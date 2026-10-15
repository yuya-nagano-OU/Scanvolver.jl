function _scan_moment_phase_q_kernel!(
    phase_q,
    alpha,
    offsets,
    q_count::Int32,
    kappa::Int32,
    max_hits::Int32,
    pixel_count::Int32,
)
    linear_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = q_count * max_hits * pixel_count
    linear_index > total && return nothing
    index = linear_index - Int32(1)
    q_index = index % q_count + Int32(1)
    index ÷= q_count
    hit_in_pixel = index % max_hits + Int32(1)
    pixel_index = index ÷ max_hits + Int32(1)

    @inbounds first_hit = offsets[pixel_index]
    @inbounds hit_count =
        offsets[pixel_index + Int32(1)] - first_hit
    if hit_in_pixel <= hit_count
        q = q_index - kappa - Int32(1)
        @inbounds phase_q[q_index, hit_in_pixel, pixel_index] =
            _scan_moment_phase(
                eltype(alpha)(q) *
                alpha[first_hit + hit_in_pixel - Int32(1)],
            )
    else
        @inbounds phase_q[q_index, hit_in_pixel, pixel_index] =
            zero(eltype(phase_q))
    end
    return nothing
end

function _scan_moment_phase_n_kernel!(
    phase_n,
    alpha,
    gamma,
    offsets,
    n_count::Int32,
    nmax::Int32,
    max_hits::Int32,
    pixel_count::Int32,
)
    linear_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = max_hits * n_count * pixel_count
    linear_index > total && return nothing
    index = linear_index - Int32(1)
    hit_in_pixel = index % max_hits + Int32(1)
    index ÷= max_hits
    n_index = index % n_count + Int32(1)
    pixel_index = index ÷ n_count + Int32(1)

    @inbounds first_hit = offsets[pixel_index]
    @inbounds hit_count =
        offsets[pixel_index + Int32(1)] - first_hit
    if hit_in_pixel <= hit_count
        hit_index = first_hit + hit_in_pixel - Int32(1)
        @inbounds alpha_plus_gamma =
            alpha[hit_index] + gamma[hit_index]
        n = n_index - nmax - Int32(1)
        @inbounds phase_n[hit_in_pixel, n_index, pixel_index] =
            _scan_moment_phase(
                eltype(alpha)(n) * alpha_plus_gamma,
            )
    else
        @inbounds phase_n[hit_in_pixel, n_index, pixel_index] =
            zero(eltype(phase_n))
    end
    return nothing
end

function _scan_moment_chebyshev_psi_kernel!(
    chebyshev,
    psi2,
    beta,
    psi,
    offsets,
    beta_order::Int32,
    beta_max::T,
    max_hits::Int32,
    pixel_count::Int32,
) where {T<:_CHEBYSHEV_PRECISIONS}
    linear_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = max_hits * pixel_count
    linear_index > total && return nothing
    hit_in_pixel =
        (linear_index - Int32(1)) % max_hits + Int32(1)
    pixel_index =
        (linear_index - Int32(1)) ÷ max_hits + Int32(1)
    @inbounds first_hit = offsets[pixel_index]
    @inbounds hit_count =
        offsets[pixel_index + Int32(1)] - first_hit
    hit_in_pixel <= hit_count || return nothing
    hit_index = first_hit + hit_in_pixel - Int32(1)
    @inbounds x = clamp(
        T(2) * beta[hit_index] / beta_max - one(T),
        -one(T),
        one(T),
    )
    @inbounds psi2[hit_in_pixel, pixel_index] =
        _scan_moment_phase(T(2) * psi[hit_index])
    @inbounds for beta_index in Int32(1):beta_order
        chebyshev[beta_index, hit_in_pixel, pixel_index] =
            _scan_moment_chebyshev_value(
                beta_index - Int32(1),
                x,
            )
    end
    return nothing
end

function _build_scan_moment_gemm_rhs_kernel!(
    rhs,
    phase_n,
    chebyshev,
    psi2,
    offsets,
    beta_index::Int32,
    spin_weight_index::Int32,
    max_hits::Int32,
    n_count::Int32,
    pixel_count::Int32,
)
    linear_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = max_hits * n_count * pixel_count
    linear_index > total && return nothing
    index = linear_index - Int32(1)
    hit_in_pixel = index % max_hits + Int32(1)
    index ÷= max_hits
    n_index = index % n_count + Int32(1)
    pixel_index = index ÷ n_count + Int32(1)

    @inbounds hit_count =
        offsets[pixel_index + Int32(1)] - offsets[pixel_index]
    if hit_in_pixel <= hit_count
        weight =
            chebyshev[beta_index, hit_in_pixel, pixel_index] /
            eltype(chebyshev)(hit_count)
        spin_phase = spin_weight_index == Int32(1) ?
            one(eltype(rhs)) :
            psi2[hit_in_pixel, pixel_index]
        @inbounds rhs[hit_in_pixel, n_index, pixel_index] =
            weight *
            phase_n[hit_in_pixel, n_index, pixel_index] *
            spin_phase
    else
        @inbounds rhs[hit_in_pixel, n_index, pixel_index] =
            zero(eltype(rhs))
    end
    return nothing
end

function _copy_scan_moment_gemm_product_kernel!(
    output,
    product,
    beta_index::Int32,
    spin_weight_index::Int32,
    q_count::Int32,
    n_count::Int32,
    pixel_count::Int32,
)
    linear_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = q_count * n_count * pixel_count
    linear_index > total && return nothing
    index = linear_index - Int32(1)
    q_index = index % q_count + Int32(1)
    index ÷= q_count
    n_index = index % n_count + Int32(1)
    pixel_index = index ÷ n_count + Int32(1)
    @inbounds output[
        spin_weight_index,
        beta_index,
        q_index,
        n_index,
        pixel_index,
    ] = product[q_index, n_index, pixel_index]
    return nothing
end

function _scan_moments_gemm_shape(
    plan::ScanMomentPlan,
    angles::AbstractVector{<:Angles},
)
    isempty(angles) && throw(ArgumentError(
        "angles must contain at least one pixel",
    ))
    for angle in angles
        _check_scan_moment_angles(plan, angle)
    end
    hit_counts = Int[angle.num for angle in angles]
    max_hits = maximum(hit_counts)
    max_hits > 0 || throw(ArgumentError(
        "GEMM scan moments require at least one hit",
    ))
    q_count, n_count, _ =
        _check_scan_moment_gpu_dimensions(plan, length(angles))
    padded_dimensions = (
        q_count,
        n_count,
        plan.beta_order,
        max_hits,
        length(angles),
    )
    all(dimension -> dimension <= typemax(Int32), padded_dimensions) ||
        throw(ArgumentError(
            "GEMM scan moment dimensions exceed Int32 indexing",
        ))
    return q_count, n_count, max_hits, length(angles)
end

"""
    scan_moments_gemm_memory_bytes(plan, angles) -> Int128

Return the estimated peak device bytes required by
`scan_moments_gemm_gpu`, including the returned output, packed angles, and
padded GEMM workspaces.
"""
function scan_moments_gemm_memory_bytes(
    plan::ScanMomentPlan{T},
    angles::AbstractVector{<:Angles},
) where {T<:_CHEBYSHEV_PRECISIONS}
    q_count, n_count, max_hits, pixel_count =
        _scan_moments_gemm_shape(plan, angles)
    total_hits = sum(Int128(angle.num) for angle in angles)
    real_bytes = Int128(sizeof(T))
    complex_bytes = Int128(sizeof(Complex{T}))
    int32_bytes = Int128(sizeof(Int32))
    padded_hits = Int128(max_hits) * pixel_count

    packed_bytes =
        Int128(4) * total_hits * real_bytes +
        (Int128(pixel_count) + 1) * int32_bytes
    output_bytes =
        Int128(2) * plan.beta_order * q_count * n_count *
        pixel_count * complex_bytes
    workspace_bytes =
        Int128(q_count) * padded_hits * complex_bytes +
        padded_hits * n_count * complex_bytes +
        Int128(plan.beta_order) * padded_hits * real_bytes +
        padded_hits * complex_bytes +
        padded_hits * n_count * complex_bytes +
        Int128(q_count) * n_count * pixel_count * complex_bytes
    return packed_bytes + output_bytes + workspace_bytes
end

"""
    scan_moments_gemm_gpu(plan, angles) -> CuArray{Complex{T},5}

Construct scan moments with reusable phase tables and one strided batched
GEMM per `(scan_weight, beta_mode)`. Pixels are padded to the largest hit
count in the supplied batch. This explicit entry point does not fall back;
use `scan_moments_gpu` for automatic memory-aware recovery.
"""
function scan_moments_gemm_gpu(
    plan::ScanMomentPlan{T},
    angles::AbstractVector{<:Angles},
) where {T<:_CHEBYSHEV_PRECISIONS}
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    q_count, n_count, max_hits, pixel_count =
        _scan_moments_gemm_shape(plan, angles)

    packed = nothing
    phase_q = nothing
    phase_n = nothing
    chebyshev = nothing
    psi2 = nothing
    rhs = nothing
    product = nothing
    output = nothing
    completed = false
    try
        packed = _PackedScanMomentAngles(angles, T)
        phase_q = CUDA.zeros(
            Complex{T},
            q_count,
            max_hits,
            pixel_count,
        )
        phase_n = CUDA.zeros(
            Complex{T},
            max_hits,
            n_count,
            pixel_count,
        )
        chebyshev = CUDA.zeros(
            T,
            plan.beta_order,
            max_hits,
            pixel_count,
        )
        psi2 = CUDA.zeros(Complex{T}, max_hits, pixel_count)
        rhs = CUDA.zeros(
            Complex{T},
            max_hits,
            n_count,
            pixel_count,
        )
        product = CUDA.zeros(
            Complex{T},
            q_count,
            n_count,
            pixel_count,
        )
        output = CUDA.zeros(
            Complex{T},
            2,
            plan.beta_order,
            q_count,
            n_count,
            pixel_count,
        )

        phase_q_total = q_count * max_hits * pixel_count
        phase_n_total = max_hits * n_count * pixel_count
        hit_total = max_hits * pixel_count
        copy_total = q_count * n_count * pixel_count
        @cuda threads=_SCAN_MOMENT_GPU_THREADS blocks=cld(
            phase_q_total,
            _SCAN_MOMENT_GPU_THREADS,
        ) _scan_moment_phase_q_kernel!(
            phase_q,
            packed.alpha,
            packed.offsets,
            Int32(q_count),
            Int32(plan.kappa),
            Int32(max_hits),
            packed.pixel_count,
        )
        @cuda threads=_SCAN_MOMENT_GPU_THREADS blocks=cld(
            phase_n_total,
            _SCAN_MOMENT_GPU_THREADS,
        ) _scan_moment_phase_n_kernel!(
            phase_n,
            packed.alpha,
            packed.gamma,
            packed.offsets,
            Int32(n_count),
            Int32(plan.nmax),
            Int32(max_hits),
            packed.pixel_count,
        )
        @cuda threads=_SCAN_MOMENT_GPU_THREADS blocks=cld(
            hit_total,
            _SCAN_MOMENT_GPU_THREADS,
        ) _scan_moment_chebyshev_psi_kernel!(
            chebyshev,
            psi2,
            packed.beta,
            packed.psi,
            packed.offsets,
            Int32(plan.beta_order),
            plan.beta_max,
            Int32(max_hits),
            packed.pixel_count,
        )

        for spin_weight_index in 1:2
            for beta_index in 1:plan.beta_order
                @cuda threads=_SCAN_MOMENT_GPU_THREADS blocks=cld(
                    phase_n_total,
                    _SCAN_MOMENT_GPU_THREADS,
                ) _build_scan_moment_gemm_rhs_kernel!(
                    rhs,
                    phase_n,
                    chebyshev,
                    psi2,
                    packed.offsets,
                    Int32(beta_index),
                    Int32(spin_weight_index),
                    Int32(max_hits),
                    Int32(n_count),
                    packed.pixel_count,
                )
                CUDA.CUBLAS.gemm_strided_batched!(
                    'N',
                    'N',
                    one(Complex{T}),
                    phase_q,
                    rhs,
                    zero(Complex{T}),
                    product,
                )
                @cuda threads=_SCAN_MOMENT_GPU_THREADS blocks=cld(
                    copy_total,
                    _SCAN_MOMENT_GPU_THREADS,
                ) _copy_scan_moment_gemm_product_kernel!(
                    output,
                    product,
                    Int32(beta_index),
                    Int32(spin_weight_index),
                    Int32(q_count),
                    Int32(n_count),
                    packed.pixel_count,
                )
            end
        end
        CUDA.synchronize()
        completed = true
        return output
    finally
        _unsafe_free_if_allocated(phase_q)
        _unsafe_free_if_allocated(phase_n)
        _unsafe_free_if_allocated(chebyshev)
        _unsafe_free_if_allocated(psi2)
        _unsafe_free_if_allocated(rhs)
        _unsafe_free_if_allocated(product)
        packed === nothing ||
            _free_packed_scan_moment_angles!(packed)
        completed || _unsafe_free_if_allocated(output)
    end
end

"""
    scan_moments_gpu(
        plan, angles;
        method=:gemm,
        fallback_to_direct=true,
        memory_fraction=0.8,
    )

Build scan moments on the GPU. GEMM is the default. Before allocation, the
estimated GEMM peak memory is compared with `memory_fraction` of currently
available device memory. If the estimate is too large, or a CUDA out-of-
memory exception occurs, `fallback_to_direct=true` emits a warning and
retries with `scan_moments_direct_gpu`.

Set `method=:direct` to force the low-workspace reference kernel, or set
`fallback_to_direct=false` to make GEMM memory failures explicit.
"""
function scan_moments_gpu(
    plan::ScanMomentPlan{T},
    angles::AbstractVector{<:Angles};
    method::Symbol=:gemm,
    fallback_to_direct::Bool=true,
    memory_fraction::Real=0.8,
) where {T<:_CHEBYSHEV_PRECISIONS}
    method in (:gemm, :direct) || throw(ArgumentError(
        "method must be :gemm or :direct; got $method",
    ))
    isfinite(memory_fraction) && 0 <= memory_fraction <= 1 ||
        throw(ArgumentError(
            "memory_fraction must be finite and in [0, 1]",
        ))
    method === :direct && return scan_moments_direct_gpu(
        plan,
        angles,
    )

    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    isempty(angles) && throw(ArgumentError(
        "angles must contain at least one pixel",
    ))
    for angle in angles
        _check_scan_moment_angles(plan, angle)
    end
    all_zero_hits = all(iszero(angle.num) for angle in angles)
    if all_zero_hits
        fallback_to_direct || throw(ArgumentError(
            "GEMM scan moments require at least one hit",
        ))
        @warn "GEMM scan moments received no hits; falling back to direct GPU path"
        return scan_moments_direct_gpu(plan, angles)
    end

    required_bytes = scan_moments_gemm_memory_bytes(plan, angles)
    available_bytes = _available_device_bytes()
    budget_bytes = floor(
        Int128,
        Float64(memory_fraction) * Float64(available_bytes),
    )
    if required_bytes > budget_bytes
        fallback_to_direct || throw(ArgumentError(
            "estimated GEMM memory $required_bytes bytes exceeds " *
            "the configured budget $budget_bytes bytes",
        ))
        @warn "GEMM scan moments exceed the memory budget; falling back to direct GPU path" required_bytes budget_bytes
        return scan_moments_direct_gpu(plan, angles)
    end

    try
        return scan_moments_gemm_gpu(plan, angles)
    catch error
        if fallback_to_direct && error isa CUDA.OutOfGPUMemoryError
            CUDA.reclaim()
            @warn "GEMM scan moments ran out of GPU memory; falling back to direct GPU path" exception=(error, catch_backtrace())
            return scan_moments_direct_gpu(plan, angles)
        end
        rethrow()
    end
end
