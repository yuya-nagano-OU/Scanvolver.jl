const _GPU_THREADS = 128
const _GPU_RECURSION_PRECISIONS = Union{Float32,Float64}

struct _RingRecursionGPUPlan{T<:_GPU_RECURSION_PRECISIONS}
    m::CuVector{Int32}
    M::CuVector{Int32}
    l0::CuVector{Int32}
    sign::CuVector{T}
    log_coefficient::CuVector{T}
    pow_c::CuVector{Int32}
    pow_s::CuVector{Int32}
end

function _check_gpu_recursion_precision(precision::Type)
    precision in (Float32, Float64) || throw(ArgumentError(
        "precision must be Float32 or Float64; got $precision",
    ))
    return precision
end

@inline _gpu_rec_scale_bits(::Type{Float32}) = Int32(60)
@inline _gpu_rec_scale_bits(::Type{Float64}) = Int32(450)

@inline function _gpu_normalize_scaled_pair(
    x::T,
    exponent::Int32,
) where {T<:_GPU_RECURSION_PRECISIONS}
    iszero(x) && return zero(T), Int32(0)

    magnitude = abs(x)
    scale_bits = _gpu_rec_scale_bits(T)
    scale = ldexp(one(T), scale_bits)
    inverse_scale = ldexp(one(T), -scale_bits)
    while magnitude > scale
        x *= inverse_scale
        exponent += Int32(1)
        magnitude = abs(x)
    end
    while magnitude < inverse_scale
        x *= scale
        exponent -= Int32(1)
        magnitude = abs(x)
    end
    return x, exponent
end

@inline function _gpu_align_scaled_value(
    x::T,
    source_exponent::Int32,
    target_exponent::Int32,
) where {T<:_GPU_RECURSION_PRECISIONS}
    (iszero(x) ||
     !isfinite(x) ||
     source_exponent == target_exponent) && return x
    source_exponent <= typemax(Int32) - Int32(4) &&
        target_exponent >= source_exponent + Int32(4) &&
        return copysign(zero(T), x)
    return ldexp(
        x,
        _gpu_rec_scale_bits(T) *
        (source_exponent - target_exponent),
    )
end

@inline function _gpu_align_scaled_terms(
    x::T,
    x_exponent::Int32,
    y::T,
    y_exponent::Int32,
) where {T<:_GPU_RECURSION_PRECISIONS}
    iszero(x) && return x, y, y_exponent
    iszero(y) && return x, y, x_exponent
    exponent = max(x_exponent, y_exponent)
    return _gpu_align_scaled_value(x, x_exponent, exponent),
        _gpu_align_scaled_value(y, y_exponent, exponent),
        exponent
end

@inline function _gpu_scaled_to_float(
    x::T,
    exponent::Int32,
) where {T<:_GPU_RECURSION_PRECISIONS}
    iszero(x) && return zero(T)
    !isfinite(x) && return x
    exponent >= Int32(4) && return copysign(T(Inf), x)
    exponent <= -Int32(4) && return copysign(zero(T), x)
    value = ldexp(x, _gpu_rec_scale_bits(T) * exponent)
    abs(value) < floatmin(T) && return copysign(zero(T), x)
    return value
end

@inline function _gpu_lmin_seed_scaled(
    initial_sign::T,
    log_coefficient::T,
    pow_c::Int32,
    pow_s::Int32,
    cosine_half::T,
    sine_half::T,
) where {T<:_GPU_RECURSION_PRECISIONS}
    sign = initial_sign
    cosine_half < zero(T) && isodd(pow_c) && (sign = -sign)
    sine_half < zero(T) && isodd(pow_s) && (sign = -sign)

    if (iszero(cosine_half) && pow_c > 0) ||
        (iszero(sine_half) && pow_s > 0)
        return zero(T), Int32(0)
    end

    logabs = log_coefficient
    pow_c > 0 &&
        (logabs += T(pow_c) * log(abs(cosine_half)))
    pow_s > 0 &&
        (logabs += T(pow_s) * log(abs(sine_half)))
    log_scale = T(_gpu_rec_scale_bits(T)) * log(T(2))
    exponent = Int32(floor(logabs / log_scale))
    x = sign * exp(logabs - T(exponent) * log_scale)
    return _gpu_normalize_scaled_pair(x, exponent)
end

@inline function _gpu_l_step_coefficients(
    l::Int32,
    m::Int32,
    n::Int32,
    ::Type{T},
) where {T<:_GPU_RECURSION_PRECISIONS}
    ll = T(l)
    mm = T(m)
    nn = T(n)
    lp1 = ll + one(T)
    lp1_squared = lp1 * lp1
    l_squared = ll * ll

    denominator = sqrt(
        (lp1_squared - mm * mm) *
        (lp1_squared - nn * nn),
    )
    f0 = lp1 * (T(2) * ll + one(T)) / denominator
    l == Int32(0) && return f0, zero(T), zero(T)

    f1 = (mm * nn) / (ll * lp1)
    f2 = (lp1 / ll) * sqrt(
        ((l_squared - mm * mm) * (l_squared - nn * nn)) /
        ((lp1_squared - mm * mm) * (lp1_squared - nn * nn)),
    )
    return f0, f1, f2
end

function _build_ring_gpu_plan(
    cp,
    ::Type{T}=Float64,
) where {T<:_GPU_RECURSION_PRECISIONS}
    m_values = Int32[]
    M_values = Int32[]
    l0_values = Int32[]
    signs = T[]
    log_coefficients = T[]
    pow_cs = Int32[]
    pow_ss = Int32[]

    Mmax = calc_Mmax(cp, cp.lstop)
    for M in 0:Mmax
        m_range = M == 0 ? (0:cp.lstop) : (-cp.lstop:cp.lstop)
        for m in m_range
            l0 = max(abs(m), M)
            seed = _precompute_wignerd_lmin_seed_plan(l0, m, M)
            push!(m_values, Int32(m))
            push!(M_values, Int32(M))
            push!(l0_values, Int32(l0))
            push!(signs, T(seed.sign))
            push!(log_coefficients, T(seed.log_coefficient))
            push!(pow_cs, Int32(seed.pow_c))
            push!(pow_ss, Int32(seed.pow_s))
        end
    end

    return _RingRecursionGPUPlan(
        CuArray(m_values),
        CuArray(M_values),
        CuArray(l0_values),
        CuArray(signs),
        CuArray(log_coefficients),
        CuArray(pow_cs),
        CuArray(pow_ss),
    )
end

function _ringd_recursion_kernel!(
    ringd,
    pair_m,
    pair_M,
    pair_l0,
    seed_sign,
    seed_log_coefficient,
    seed_pow_c,
    seed_pow_s,
    beta::T,
    lstart::Int32,
    lstop::Int32,
    Mmax_global::Int32,
    pair_count::Int32,
) where {T<:_GPU_RECURSION_PRECISIONS}
    pair_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    pair_index > pair_count && return nothing

    @inbounds m = pair_m[pair_index]
    @inbounds M = pair_M[pair_index]
    @inbounds l0 = pair_l0[pair_index]
    cosine_beta = cos(beta)
    cosine_half = cos(T(0.5) * beta)
    sine_half = sin(T(0.5) * beta)
    previous_x = zero(T)
    previous_exponent = Int32(0)
    @inbounds current_x, current_exponent = _gpu_lmin_seed_scaled(
        seed_sign[pair_index],
        seed_log_coefficient[pair_index],
        seed_pow_c[pair_index],
        seed_pow_s[pair_index],
        cosine_half,
        sine_half,
    )

    if l0 >= lstart
        Mmax_l = min(Mmax_global, l0)
        l_index = l0 - lstart + Int32(1)
        row = m + l0 + Int32(1)
        column = M + Mmax_l + Int32(1)
        value = _gpu_scaled_to_float(current_x, current_exponent)
        @inbounds ringd[row, column, l_index] =
            Complex{T}(value, zero(T))
        if !(m == Int32(0) && M == Int32(0))
            mirror_row = -m + l0 + Int32(1)
            mirror_column = -M + Mmax_l + Int32(1)
            mirror_value = isodd(m - M) ? -value : value
            @inbounds ringd[
                mirror_row,
                mirror_column,
                l_index,
            ] = Complex{T}(mirror_value, zero(T))
        end
    end

    for l in l0:(lstop - Int32(1))
        f0, f1, f2 = _gpu_l_step_coefficients(l, m, M, T)
        current_aligned, previous_aligned, exponent =
            _gpu_align_scaled_terms(
                current_x,
                current_exponent,
                previous_x,
                previous_exponent,
            )
        next_x =
            f0 * (cosine_beta - f1) * current_aligned -
            f2 * previous_aligned
        next_x, next_exponent =
            _gpu_normalize_scaled_pair(next_x, exponent)
        previous_x, previous_exponent =
            current_x, current_exponent
        current_x, current_exponent = next_x, next_exponent

        next_l = l + Int32(1)
        next_l < lstart && continue
        Mmax_l = min(Mmax_global, next_l)
        l_index = next_l - lstart + Int32(1)
        row = m + next_l + Int32(1)
        column = M + Mmax_l + Int32(1)
        value = _gpu_scaled_to_float(current_x, current_exponent)
        @inbounds ringd[row, column, l_index] =
            Complex{T}(value, zero(T))
        if !(m == Int32(0) && M == Int32(0))
            mirror_row = -m + next_l + Int32(1)
            mirror_column = -M + Mmax_l + Int32(1)
            mirror_value = isodd(m - M) ? -value : value
            @inbounds ringd[
                mirror_row,
                mirror_column,
                l_index,
            ] = Complex{T}(mirror_value, zero(T))
        end
    end
    return nothing
end

function _compute_ringd_recursion_gpu!(
    ringd,
    plan::_RingRecursionGPUPlan{T},
    cp,
    beta::Real,
) where {T<:_GPU_RECURSION_PRECISIONS}
    fill!(ringd, zero(Complex{T}))
    pair_count = length(plan.m)
    blocks = cld(pair_count, _GPU_THREADS)
    @cuda threads=_GPU_THREADS blocks=blocks _ringd_recursion_kernel!(
        ringd,
        plan.m,
        plan.M,
        plan.l0,
        plan.sign,
        plan.log_coefficient,
        plan.pow_c,
        plan.pow_s,
        T(beta),
        Int32(cp.lstart),
        Int32(cp.lstop),
        Int32(calc_Mmax(cp, cp.lstop)),
        Int32(pair_count),
    )
    return ringd
end

"""
    ringd_blocks_recursion_gpu(
        cp, beta; device_id=nothing, precision=Float64,
    )

Compute the complete ringD stack on the selected CUDA device. The padded
CuArray has layout (m, M, l) and shape
(2lstop+1, 2Mmax(lstop)+1, length(lstart:lstop)).
"""
function ringd_blocks_recursion_gpu(
    cp,
    beta::Real;
    device_id::Union{Nothing,Integer}=nothing,
    precision::Type=Float64,
)
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    device_id === nothing || CUDA.device!(Int(device_id))
    T = _check_gpu_recursion_precision(precision)
    plan = _build_ring_gpu_plan(cp, T)
    ringd = CUDA.zeros(
        Complex{T},
        2 * cp.lstop + 1,
        2 * calc_Mmax(cp, cp.lstop) + 1,
        cp.lstop - cp.lstart + 1,
    )
    return _compute_ringd_recursion_gpu!(ringd, plan, cp, beta)
end

"""
    RingDContext{T}

The reusable half of the ringD computation: the seed plan and the output
buffer.

`ringd_blocks_recursion_gpu` builds both on every call. The seed plan is one
entry per `(m, M)` pair, which is about 627k pairs at lmax 1200 and Mmax 130,
each assembled on the host and uploaded; the buffer is 5.6 GB at that size in
Float32. Neither depends on the ring, only on `lstart`, `lstop`, `Mmax` and the
precision, so a full sphere should build them once and refill the buffer per
ring.
"""
struct RingDContext{T<:_GPU_RECURSION_PRECISIONS}
    plan::_RingRecursionGPUPlan{T}
    ringd::CuArray{Complex{T},3}
    lstart::Int
    lstop::Int
    Mmax::Int
end

Base.eltype(::RingDContext{T}) where {T} = Complex{T}

function Base.show(io::IO, context::RingDContext{T}) where {T}
    print(
        io,
        "RingDContext{$T}(l=$(context.lstart):$(context.lstop), ",
        "Mmax=$(context.Mmax), ",
        "$(round(sizeof(context.ringd) / 2^20; digits=1)) MiB)",
    )
end

"""
    ringd_context(cp; precision=Float64, device_id=nothing) -> RingDContext

Allocate the seed plan and the ringD buffer once for a whole sphere. Release it
with `free_ringd_context!`.
"""
function ringd_context(
    cp;
    precision::Type=Float64,
    device_id::Union{Nothing,Integer}=nothing,
)
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    T = _check_gpu_recursion_precision(precision)
    device_id === nothing || CUDA.device!(Int(device_id))
    Mmax = calc_Mmax(cp, cp.lstop)
    plan = _build_ring_gpu_plan(cp, T)
    ringd = nothing
    succeeded = false
    try
        ringd = CUDA.zeros(
            Complex{T},
            2 * cp.lstop + 1,
            2 * Mmax + 1,
            cp.lstop - cp.lstart + 1,
        )
        succeeded = true
        return RingDContext{T}(plan, ringd, cp.lstart, cp.lstop, Mmax)
    finally
        if !succeeded
            ringd === nothing || CUDA.unsafe_free!(ringd)
            _free_ring_gpu_plan!(plan)
        end
    end
end

function _free_ring_gpu_plan!(plan::_RingRecursionGPUPlan)
    CUDA.unsafe_free!(plan.m)
    CUDA.unsafe_free!(plan.M)
    CUDA.unsafe_free!(plan.l0)
    CUDA.unsafe_free!(plan.sign)
    CUDA.unsafe_free!(plan.log_coefficient)
    CUDA.unsafe_free!(plan.pow_c)
    CUDA.unsafe_free!(plan.pow_s)
    return nothing
end

"""
    free_ringd_context!(context)

Release the seed plan and the ringD buffer.
"""
function free_ringd_context!(context::RingDContext)
    CUDA.unsafe_free!(context.ringd)
    _free_ring_gpu_plan!(context.plan)
    return nothing
end

function _check_ringd_context(context::RingDContext, cp)
    context.lstart == cp.lstart && context.lstop == cp.lstop ||
        throw(ArgumentError(
            "RingDContext covers l=$(context.lstart):$(context.lstop); " *
            "ConvolutionParams covers $(cp.lstart):$(cp.lstop)",
        ))
    context.Mmax == calc_Mmax(cp, cp.lstop) || throw(ArgumentError(
        "RingDContext has Mmax=$(context.Mmax); ConvolutionParams gives " *
        "$(calc_Mmax(cp, cp.lstop))",
    ))
    return nothing
end

"""
    ringd_blocks_recursion_gpu!(context, cp, beta) -> CuArray

Refill a `RingDContext` for one ring colatitude and return its buffer. The
buffer is owned by the context, so it must not be freed by the caller and its
contents are replaced by the next call.
"""
function ringd_blocks_recursion_gpu!(
    context::RingDContext{T},
    cp,
    beta::Real,
) where {T<:_GPU_RECURSION_PRECISIONS}
    _check_ringd_context(context, cp)
    return _compute_ringd_recursion_gpu!(
        context.ringd,
        context.plan,
        cp,
        beta,
    )
end
