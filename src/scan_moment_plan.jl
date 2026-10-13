const _CHEBYSHEV_PRECISIONS = Union{Float32,Float64}

"""
    ScanMomentPlan(; kappa, nmax, beta_order, beta_max, precision=Float32)

Configuration for a Fourier--Chebyshev scan representation. `beta_order`
sets the number of first-kind Chebyshev nodes on `[0, beta_max]`.
"""
struct ScanMomentPlan{T<:_CHEBYSHEV_PRECISIONS}
    kappa::Int
    nmax::Int
    beta_order::Int
    beta_max::T
end

function ScanMomentPlan(;
    kappa::Integer,
    nmax::Integer,
    beta_order::Integer,
    beta_max::Real,
    precision::Type=Float32,
)
    precision in (Float32, Float64) || throw(ArgumentError(
        "precision must be Float32 or Float64; got $precision",
    ))
    kappa >= 0 || throw(ArgumentError("kappa must be non-negative"))
    nmax >= 0 || throw(ArgumentError("nmax must be non-negative"))
    beta_order > 0 || throw(ArgumentError("beta_order must be positive"))
    isfinite(beta_max) && beta_max > 0 || throw(ArgumentError(
        "beta_max must be finite and positive",
    ))

    T = precision
    return ScanMomentPlan{T}(
        Int(kappa),
        Int(nmax),
        Int(beta_order),
        T(beta_max),
    )
end

"""
    chebyshev_nodes(plan)

Return first-kind Chebyshev nodes mapped from `[-1, 1]` to
`[0, plan.beta_max]`. The root grid excludes both interval endpoints.
"""
function chebyshev_nodes(plan::ScanMomentPlan{T}) where {T}
    order = plan.beta_order
    half = plan.beta_max / T(2)
    return T[
        half * (one(T) + cos(T(pi) * T(2j - 1) / T(2order)))
        for j in 1:order
    ]
end

@inline function _scaled_beta(plan::ScanMomentPlan{T}, beta::Real) where {T}
    tolerance = T(32) * eps(T) * max(one(T), plan.beta_max)
    beta >= -tolerance && beta <= plan.beta_max + tolerance ||
        throw(DomainError(
            beta,
            "beta is outside the Chebyshev interval [0,$(plan.beta_max)]",
        ))
    clipped = clamp(T(beta), zero(T), plan.beta_max)
    return T(2) * clipped / plan.beta_max - one(T)
end

function _chebyshev_values!(
    values::AbstractVector{T},
    plan::ScanMomentPlan{T},
    beta::Real,
) where {T}
    length(values) == plan.beta_order || throw(DimensionMismatch(
        "Chebyshev workspace has length $(length(values)); " *
        "expected $(plan.beta_order)",
    ))
    x = _scaled_beta(plan, beta)
    values[1] = one(T)
    plan.beta_order == 1 && return values
    values[2] = x
    @inbounds for index in 3:plan.beta_order
        values[index] =
            T(2) * x * values[index - 1] - values[index - 2]
    end
    return values
end

function _chebyshev_transform(plan::ScanMomentPlan{T}) where {T}
    order = plan.beta_order
    transform = Matrix{T}(undef, order, order)
    @inbounds for r_index in 1:order
        r = r_index - 1
        scale = r == 0 ? inv(T(order)) : T(2) / T(order)
        for node_index in 1:order
            theta = T(pi) * T(2node_index - 1) / T(2order)
            transform[r_index, node_index] =
                scale * cos(T(r) * theta)
        end
    end
    return transform
end
