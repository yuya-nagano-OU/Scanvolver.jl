const _SCAN_MOMENT_GPU_THREADS = 256

struct _PackedScanMomentAngles{T<:_CHEBYSHEV_PRECISIONS}
    alpha::CuVector{T}
    beta::CuVector{T}
    gamma::CuVector{T}
    psi::CuVector{T}
    offsets::CuVector{Int32}
    pixel_count::Int32
end

function _unsafe_free_if_allocated(array)
    array === nothing || CUDA.unsafe_free!(array)
    return nothing
end

function _PackedScanMomentAngles(
    angles::AbstractVector{<:Angles},
    ::Type{T},
) where {T<:_CHEBYSHEV_PRECISIONS}
    isempty(angles) && throw(ArgumentError(
        "angles must contain at least one pixel",
    ))
    length(angles) <= typemax(Int32) || throw(ArgumentError(
        "pixel count exceeds Int32 indexing",
    ))
    total_hits_wide = sum(Int128(angle.num) for angle in angles)
    total_hits_wide < typemax(Int32) || throw(ArgumentError(
        "packed angle batch exceeds Int32 indexing",
    ))
    total_hits = Int(total_hits_wide)

    alpha = Vector{T}(undef, total_hits)
    beta = Vector{T}(undef, total_hits)
    gamma = Vector{T}(undef, total_hits)
    psi = Vector{T}(undef, total_hits)
    offsets = Vector{Int32}(undef, length(angles) + 1)
    offsets[1] = Int32(1)

    position = 1
    @inbounds for pixel_index in eachindex(angles)
        angle = angles[pixel_index]
        hit_count = angle.num
        destination = position:(position + hit_count - 1)
        alpha[destination] .= T.(angle.α)
        beta[destination] .= T.(angle.β)
        gamma[destination] .= T.(angle.γ)
        psi[destination] .= T.(angle.ψ)
        position += hit_count
        offsets[pixel_index + 1] = Int32(position)
    end

    alpha_device = nothing
    beta_device = nothing
    gamma_device = nothing
    psi_device = nothing
    offsets_device = nothing
    completed = false
    try
        alpha_device = CuArray(alpha)
        beta_device = CuArray(beta)
        gamma_device = CuArray(gamma)
        psi_device = CuArray(psi)
        offsets_device = CuArray(offsets)
        packed = _PackedScanMomentAngles{T}(
            alpha_device,
            beta_device,
            gamma_device,
            psi_device,
            offsets_device,
            Int32(length(angles)),
        )
        completed = true
        return packed
    finally
        if !completed
            _unsafe_free_if_allocated(alpha_device)
            _unsafe_free_if_allocated(beta_device)
            _unsafe_free_if_allocated(gamma_device)
            _unsafe_free_if_allocated(psi_device)
            _unsafe_free_if_allocated(offsets_device)
        end
    end
end

function _free_packed_scan_moment_angles!(
    packed::_PackedScanMomentAngles,
)
    CUDA.unsafe_free!(packed.alpha)
    CUDA.unsafe_free!(packed.beta)
    CUDA.unsafe_free!(packed.gamma)
    CUDA.unsafe_free!(packed.psi)
    CUDA.unsafe_free!(packed.offsets)
    return nothing
end

@inline function _scan_moment_chebyshev_value(
    order::Int32,
    x::T,
) where {T<:_CHEBYSHEV_PRECISIONS}
    order == Int32(0) && return one(T)
    order == Int32(1) && return x
    previous = one(T)
    current = x
    for _ in Int32(2):order
        next = T(2) * x * current - previous
        previous = current
        current = next
    end
    return current
end

@inline function _scan_moment_phase(angle::T) where
    {T<:_CHEBYSHEV_PRECISIONS}
    return Complex{T}(cos(angle), sin(angle))
end

function _scan_moments_direct_kernel!(
    output,
    alpha,
    beta,
    gamma,
    psi,
    offsets,
    beta_order::Int32,
    q_count::Int32,
    n_count::Int32,
    kappa::Int32,
    nmax::Int32,
    beta_max::T,
    pixel_count::Int32,
) where {T<:_CHEBYSHEV_PRECISIONS}
    linear_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = beta_order * q_count * n_count * pixel_count
    linear_index > total && return nothing

    index = linear_index - Int32(1)
    beta_index = index % beta_order + Int32(1)
    index ÷= beta_order
    q_index = index % q_count + Int32(1)
    index ÷= q_count
    n_index = index % n_count + Int32(1)
    pixel_index = index ÷ n_count + Int32(1)

    @inbounds first_hit = offsets[pixel_index]
    @inbounds stop_hit = offsets[pixel_index + Int32(1)] - Int32(1)
    hit_count = stop_hit - first_hit + Int32(1)
    hit_count <= Int32(0) && return nothing
    inverse_hits = inv(T(hit_count))
    q = q_index - kappa - Int32(1)
    n = n_index - nmax - Int32(1)
    beta_mode = beta_index - Int32(1)
    spin0 = zero(Complex{T})
    spin2 = zero(Complex{T})

    @inbounds for hit_index in first_hit:stop_hit
        alpha_value = alpha[hit_index]
        alpha_plus_gamma = alpha_value + gamma[hit_index]
        spin0_phase = _scan_moment_phase(
            T(q) * alpha_value + T(n) * alpha_plus_gamma,
        )
        x = clamp(
            T(2) * beta[hit_index] / beta_max - one(T),
            -one(T),
            one(T),
        )
        weight =
            _scan_moment_chebyshev_value(beta_mode, x) *
            inverse_hits
        spin0 += weight * spin0_phase
        spin2 +=
            weight * spin0_phase *
            _scan_moment_phase(T(2) * psi[hit_index])
    end

    @inbounds output[
        Int32(1),
        beta_index,
        q_index,
        n_index,
        pixel_index,
    ] = spin0
    @inbounds output[
        Int32(2),
        beta_index,
        q_index,
        n_index,
        pixel_index,
    ] = spin2
    return nothing
end

function _check_scan_moment_gpu_dimensions(
    plan::ScanMomentPlan,
    pixel_count::Integer,
)
    q_count = 2 * plan.kappa + 1
    n_count = 2 * plan.nmax + 1
    dimensions = (
        plan.beta_order,
        q_count,
        n_count,
        Int(pixel_count),
    )
    all(dimension -> dimension <= typemax(Int32), dimensions) ||
        throw(ArgumentError(
            "scan moment dimensions exceed Int32 indexing",
        ))
    total = prod(Int128(dimension) for dimension in dimensions)
    total <= typemax(Int32) || throw(ArgumentError(
        "scan moment output exceeds direct-kernel Int32 indexing",
    ))
    return q_count, n_count, Int(total)
end

"""
    scan_moments_direct_gpu(plan, angles) -> CuArray{Complex{T},5}

Compute direct-sum scan moments for a vector of active pixels on the GPU.
The result has layout `(scan_weight, beta_mode, q, n, pixel)`. This is the
clear reference GPU path; optimized GEMM construction is intentionally kept
separate.
"""
function scan_moments_direct_gpu(
    plan::ScanMomentPlan{T},
    angles::AbstractVector{<:Angles},
) where {T<:_CHEBYSHEV_PRECISIONS}
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    for angle in angles
        _check_scan_moment_angles(plan, angle)
    end
    q_count, n_count, total =
        _check_scan_moment_gpu_dimensions(plan, length(angles))
    packed = _PackedScanMomentAngles(angles, T)
    output = CUDA.zeros(
        Complex{T},
        2,
        plan.beta_order,
        q_count,
        n_count,
        length(angles),
    )
    completed = false
    try
        blocks = cld(total, _SCAN_MOMENT_GPU_THREADS)
        @cuda threads=_SCAN_MOMENT_GPU_THREADS blocks=blocks _scan_moments_direct_kernel!(
            output,
            packed.alpha,
            packed.beta,
            packed.gamma,
            packed.psi,
            packed.offsets,
            Int32(plan.beta_order),
            Int32(q_count),
            Int32(n_count),
            Int32(plan.kappa),
            Int32(plan.nmax),
            plan.beta_max,
            packed.pixel_count,
        )
        CUDA.synchronize()
        completed = true
    finally
        _free_packed_scan_moment_angles!(packed)
        completed || CUDA.unsafe_free!(output)
    end
    return output
end
