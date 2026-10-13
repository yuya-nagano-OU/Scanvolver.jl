const _CPU_WIGNER_REC_SCALE = 2.0^450
const _CPU_WIGNER_REC_INV_SCALE = 2.0^-450
const _CPU_WIGNER_REC_SCALE_BITS = 450
const _CPU_WIGNER_REC_LOG_SCALE = 450.0 * log(2.0)

@inline function _cpu_normalize_scaled_pair(x::Float64, exponent::Int)
    iszero(x) && return 0.0, 0

    magnitude = abs(x)
    while magnitude > _CPU_WIGNER_REC_SCALE
        x *= _CPU_WIGNER_REC_INV_SCALE
        exponent += 1
        magnitude = abs(x)
    end
    while magnitude < _CPU_WIGNER_REC_INV_SCALE
        x *= _CPU_WIGNER_REC_SCALE
        exponent -= 1
        magnitude = abs(x)
    end
    return x, exponent
end

@inline function _cpu_align_scaled_value(
    x::Float64,
    source_exponent::Int,
    target_exponent::Int,
)
    (iszero(x) ||
     !isfinite(x) ||
     source_exponent == target_exponent) && return x
    source_exponent <= typemax(Int) - 4 &&
        target_exponent >= source_exponent + 4 &&
        return copysign(0.0, x)
    return ldexp(
        x,
        _CPU_WIGNER_REC_SCALE_BITS *
        (source_exponent - target_exponent),
    )
end

@inline function _cpu_align_scaled_terms(
    x::Float64,
    x_exponent::Int,
    y::Float64,
    y_exponent::Int,
)
    iszero(x) && return x, y, y_exponent
    iszero(y) && return x, y, x_exponent
    exponent = max(x_exponent, y_exponent)
    return _cpu_align_scaled_value(x, x_exponent, exponent),
        _cpu_align_scaled_value(y, y_exponent, exponent),
        exponent
end

@inline function _cpu_scaled_to_float(x::Float64, exponent::Int)
    iszero(x) && return 0.0
    !isfinite(x) && return x
    exponent >= 4 && return copysign(Inf, x)
    exponent <= -4 && return copysign(0.0, x)
    value = ldexp(x, _CPU_WIGNER_REC_SCALE_BITS * exponent)
    abs(value) < floatmin(Float64) && return copysign(0.0, x)
    return value
end

@inline function _cpu_lmin_seed_scaled(
    seed::_WignerdRecursionSeedPlan,
    cosine_half::Float64,
    sine_half::Float64,
)
    sign = seed.sign
    cosine_half < 0 && isodd(seed.pow_c) && (sign = -sign)
    sine_half < 0 && isodd(seed.pow_s) && (sign = -sign)

    if (iszero(cosine_half) && seed.pow_c > 0) ||
       (iszero(sine_half) && seed.pow_s > 0)
        return 0.0, 0
    end

    logabs = seed.log_coefficient
    seed.pow_c > 0 &&
        (logabs += seed.pow_c * log(abs(cosine_half)))
    seed.pow_s > 0 &&
        (logabs += seed.pow_s * log(abs(sine_half)))
    exponent = floor(Int, logabs / _CPU_WIGNER_REC_LOG_SCALE)
    x = sign * exp(
        logabs - exponent * _CPU_WIGNER_REC_LOG_SCALE,
    )
    return _cpu_normalize_scaled_pair(x, exponent)
end

@inline function _cpu_l_step_coefficients(
    l::Integer,
    M::Integer,
    n::Integer,
)
    ll = Float64(l)
    MM = Float64(M)
    nn = Float64(n)
    lp1 = ll + 1.0
    lp1_squared = lp1^2
    l_squared = ll^2

    denominator = sqrt(
        (lp1_squared - MM^2) *
        (lp1_squared - nn^2),
    )
    f0 = lp1 * (2.0 * ll + 1.0) / denominator
    iszero(l) && return f0, 0.0, 0.0

    f1 = (MM * nn) / (ll * lp1)
    f2 = (lp1 / ll) * sqrt(
        ((l_squared - MM^2) * (l_squared - nn^2)) /
        ((lp1_squared - MM^2) * (lp1_squared - nn^2)),
    )
    return f0, f1, f2
end

struct _WignerChebyshevCPUPair
    M::Int
    n::Int
    l0::Int
    first_l::Int
    seed::_WignerdRecursionSeedPlan
    f0::Vector{Float64}
    f1::Vector{Float64}
    f2::Vector{Float64}
end

function _build_wigner_chebyshev_cpu_pairs(cp, plan::ScanMomentPlan)
    pairs = _WignerChebyshevCPUPair[]
    Mmax = calc_Mmax(cp, cp.lstop)
    for n in -plan.nmax:plan.nmax
        first_M = max(-Mmax, n - plan.kappa)
        last_M = min(Mmax, n + plan.kappa)
        for M in first_M:last_M
            l0 = max(abs(M), abs(n))
            l0 > cp.lstop && continue
            step_count = cp.lstop - l0
            f0 = Vector{Float64}(undef, step_count)
            f1 = Vector{Float64}(undef, step_count)
            f2 = Vector{Float64}(undef, step_count)
            @inbounds for step_index in 1:step_count
                l = l0 + step_index - 1
                f0[step_index], f1[step_index], f2[step_index] =
                    _cpu_l_step_coefficients(l, M, n)
            end
            push!(
                pairs,
                _WignerChebyshevCPUPair(
                    M,
                    n,
                    l0,
                    max(cp.lstart, l0),
                    _precompute_wignerd_lmin_seed_plan(l0, M, n),
                    f0,
                    f1,
                    f2,
                ),
            )
        end
    end
    isempty(pairs) && throw(ArgumentError(
        "the Wigner-Chebyshev plan contains no valid (M, n) pairs",
    ))
    return pairs
end

function _accumulate_wigner_chebyshev_pair_node_cpu!(
    coefficients::Array{T,3},
    pair_index::Int,
    pair::_WignerChebyshevCPUPair,
    cp,
    beta::T,
    transform_column::AbstractVector{T},
) where {T<:_CHEBYSHEV_PRECISIONS}
    cosine_beta = cos(Float64(beta))
    cosine_half = cos(Float64(beta) / 2.0)
    sine_half = sin(Float64(beta) / 2.0)
    previous_x = 0.0
    previous_exponent = 0
    current_x, current_exponent = _cpu_lmin_seed_scaled(
        pair.seed,
        cosine_half,
        sine_half,
    )

    if pair.l0 >= pair.first_l
        value = T(_cpu_scaled_to_float(
            current_x,
            current_exponent,
        ))
        l_index = pair.l0 - cp.lstart + 1
        @inbounds for r_index in axes(coefficients, 1)
            coefficients[r_index, pair_index, l_index] +=
                transform_column[r_index] * value
        end
    end

    @inbounds for step_index in eachindex(pair.f0)
        current_aligned, previous_aligned, exponent =
            _cpu_align_scaled_terms(
                current_x,
                current_exponent,
                previous_x,
                previous_exponent,
            )
        next_x =
            pair.f0[step_index] *
            (cosine_beta - pair.f1[step_index]) *
            current_aligned -
            pair.f2[step_index] * previous_aligned
        next_x, next_exponent =
            _cpu_normalize_scaled_pair(next_x, exponent)
        previous_x, previous_exponent = current_x, current_exponent
        current_x, current_exponent = next_x, next_exponent

        l = pair.l0 + step_index
        l < pair.first_l && continue
        value = T(_cpu_scaled_to_float(
            current_x,
            current_exponent,
        ))
        l_index = l - cp.lstart + 1
        for r_index in axes(coefficients, 1)
            coefficients[r_index, pair_index, l_index] +=
                transform_column[r_index] * value
        end
    end
    return nothing
end

"""
    build_wigner_chebyshev_table_cpu(cp, plan) -> WignerChebyshevTable

Build the scan-independent Chebyshev coefficients of all required
`d^l_{Mn}(beta)` values on the CPU, with a recurrence independent of the GPU
ringD code. Send the table to a device with `upload_wigner_table_gpu`. Does not
require a functional CUDA device.
"""
function build_wigner_chebyshev_table_cpu(
    cp,
    plan::ScanMomentPlan{T},
) where {T<:_CHEBYSHEV_PRECISIONS}
    _check_wigner_chebyshev_inputs(cp, plan)
    pairs = _build_wigner_chebyshev_cpu_pairs(cp, plan)
    pair_count = length(pairs)
    l_count = cp.lstop - cp.lstart + 1
    coefficients = zeros(
        T,
        plan.beta_order,
        pair_count,
        l_count,
    )
    nodes = chebyshev_nodes(plan)
    transform = _chebyshev_transform(plan)

    Threads.@threads :greedy for pair_index in eachindex(pairs)
        pair = pairs[pair_index]
        for node_index in eachindex(nodes)
            _accumulate_wigner_chebyshev_pair_node_cpu!(
                coefficients,
                pair_index,
                pair,
                cp,
                nodes[node_index],
                @view(transform[:, node_index]),
            )
        end
    end

    return WignerChebyshevTable{T}(
        cp.lstart,
        cp.lstop,
        plan.kappa,
        plan.nmax,
        plan.beta_max,
        Int32[pair.M for pair in pairs],
        Int32[pair.n for pair in pairs],
        Int32[pair.first_l for pair in pairs],
        coefficients,
    )
end
