"""One exact-recurrence Wigner-d element; no Chebyshev interpolation or GPU."""
function _kappa_wigner_d(l::Int, M::Int, n::Int, beta::Float64)
    iszero(beta) && return M == n ? 1.0 : 0.0
    l0 = max(abs(M), abs(n))
    seed = _precompute_wignerd_lmin_seed_plan(l0, M, n)
    current, current_exponent = _cpu_lmin_seed_scaled(
        seed, cos(beta / 2.0), sin(beta / 2.0),
    )
    previous, previous_exponent = 0.0, 0
    cosine_beta = cos(beta)
    for degree in l0:(l - 1)
        f0, f1, f2 = _cpu_l_step_coefficients(degree, M, n)
        aligned, aligned_previous, exponent = _cpu_align_scaled_terms(
            current, current_exponent, previous, previous_exponent,
        )
        next_value = f0 * (cosine_beta - f1) * aligned - f2 * aligned_previous
        next_value, next_exponent = _cpu_normalize_scaled_pair(next_value, exponent)
        previous, previous_exponent = current, current_exponent
        current, current_exponent = next_value, next_exponent
    end
    value = _cpu_scaled_to_float(current, current_exponent)
    isfinite(value) || throw(ErrorException("non-finite Wigner-d value while estimating kappa"))
    return value
end

"""
    estimate_kappa(lmax, nmax, nside, threshold; kwargs...) -> Int
    estimate_kappa(lmax, nmax, nside; threshold=1e-2, kwargs...) -> Int

Estimate the smallest non-negative Fourier band `kappa` at **one endpoint**:
`l = lmax`, `beta = automatic_beta_max(nside; precision=Float64)`. Retain
`abs(M-n) <= kappa`; require every omitted component, for `abs(n) <= nmax`
and `abs(M) <= lmax`, to have magnitude at most `threshold` after normalization.

The default `normalization=:diagonal` uses
`abs(d^l_{M,n}(beta)) / abs(d^l_{n,n}(beta))`. `normalization=:absolute`
instead tests `abs(d^l_{M,n}(beta))` without dividing by the diagonal.
`threshold` is an **amplitude**, not a power or map-error tolerance.

Keyword arguments:

- `beta_max=nothing`: override the evaluation angle in radians, in `[0, pi]`.
- `normalization=:diagonal`: `:diagonal` or `:absolute`.
- `diagonal_atol=sqrt(eps(Float64))`: in diagonal mode throw `DomainError`
  if any normalization denominator is at or below this value; never silently
  clamp a near-zero diagonal or switch normalization.
- `details=false`: return an integer normally; with `true`, return a named
  tuple including `kappa`, `beta_max`, `q_values`, `ratios`, `tail_bound`,
  `minimum_diagonal`, and `omitted_maximum_bound`.

CPU-only Float64 evaluation reuses the scaled Wigner recurrence. It does not
stop at the first sub-threshold component: farther components are evaluated
or excluded with a matrix-exponential tail bound. Floating-point roundoff is
not certified by that bound. No sky/beam weights or map conditioning enter.

This is **not** a guarantee over all `l <= lmax`, over the whole angle
interval, or for final IQU maps. A diagonal can have an interior zero even
when its endpoint is nonzero. Use this as a candidate for convergence tests;
it does not alter `ConvolutionParams` or a running pipeline.

# Examples
```julia
kappa = estimate_kappa(1000, 200, 512, 1e-2)  # 5
info = estimate_kappa(1000, 200, 512, 1e-2; details=true)
info.kappa, info.beta_max, info.omitted_maximum_bound
```
"""
function estimate_kappa(
    lmax::Integer,
    nmax::Integer,
    nside::Integer,
    threshold::Real;
    beta_max::Union{Nothing,Real}=nothing,
    normalization::Symbol=:diagonal,
    diagonal_atol::Real=sqrt(eps(Float64)),
    details::Bool=false,
)
    lmax >= 0 || throw(ArgumentError("lmax must be non-negative"))
    0 <= nmax <= lmax || throw(ArgumentError("nmax must satisfy 0 <= nmax <= lmax"))
    nside > 0 || throw(ArgumentError("nside must be positive"))
    isfinite(threshold) && threshold > 0 || throw(ArgumentError(
        "threshold must be finite and positive",
    ))
    normalization in (:diagonal, :absolute) || throw(ArgumentError(
        "normalization must be :diagonal or :absolute",
    ))
    isfinite(diagonal_atol) && diagonal_atol >= 0 || throw(ArgumentError(
        "diagonal_atol must be finite and non-negative",
    ))
    l, n_limit, resolution = Int(lmax), Int(nmax), Int(nside)
    l <= typemax(Int) - n_limit || throw(ArgumentError("lmax + nmax overflows Int"))
    # Validate the HEALPix resolution even if the caller overrides the angle.
    automatic_angle = automatic_beta_max(resolution; precision=Float64)
    beta = beta_max === nothing ? automatic_angle : Float64(beta_max)
    isfinite(beta) && 0 <= beta <= Float64(pi) || throw(ArgumentError(
        "beta_max must be finite and in [0, pi] radians",
    ))
    target, diagonal_floor = Float64(threshold), Float64(diagonal_atol)
    isfinite(target) && target > 0 || throw(ArgumentError(
        "threshold must be representable as a finite positive Float64",
    ))
    isfinite(diagonal_floor) || throw(ArgumentError(
        "diagonal_atol must be representable as a finite Float64",
    ))

    # Simultaneous sign reversal of (M,n) leaves the magnitude unchanged.
    # Non-negative n plus both sides M=n +/- q therefore cover signed n too.
    diagonal = [abs(_kappa_wigner_d(l, n, n, beta)) for n in 0:n_limit]
    minimum_diagonal = minimum(diagonal)
    if normalization === :diagonal && minimum_diagonal <= diagonal_floor
        throw(DomainError(minimum_diagonal,
            "a diagonal Wigner-d normalization is too small at the evaluation angle; " *
            "change beta_max or explicitly request normalization=:absolute"))
    end
    denominator = normalization === :diagonal ? diagonal : ones(n_limit + 1)
    minimum_denominator = minimum(denominator)
    max_q = l + n_limit

    # For G=-i*beta*J_y, ||G||_2=l*beta, and (G^r)_{Mn}=0 when r<|M-n|.
    # Thus |d_{Mn}| <= sum_{r=q}^Inf x^r/r! <= exp(x)*x^q/q!.
    # Once q>x this bound decreases with q. It excludes ALL more distant
    # components, without assuming that the Wigner-d amplitudes are monotone.
    x = l * beta
    evaluated_q = max_q
    tail_bound = 0.0
    if iszero(x)
        evaluated_q = 0
    else
        log_target = log(target)
        log_denominator = log(minimum_denominator)
        for q in 1:max_q
            q > x || continue
            log_bound = x + q * log(x) - loggamma(Float64(q) + 1.0) - log_denominator
            if log_bound <= log_target
                evaluated_q = q - 1
                tail_bound = exp(log_bound)
                break
            end
        end
    end

    ratios = zeros(evaluated_q + 1)
    ratios[1] = normalization === :diagonal ? 1.0 : maximum(diagonal)
    kappa = 0
    for q in 1:evaluated_q
        largest = 0.0
        for n in 0:n_limit
            lower = n - q
            if lower >= -l
                largest = max(largest, abs(_kappa_wigner_d(l, lower, n, beta)) / denominator[n + 1])
            end
            upper = n + q
            if upper <= l
                largest = max(largest, abs(_kappa_wigner_d(l, upper, n, beta)) / denominator[n + 1])
            end
        end
        ratios[q + 1] = largest
        largest > target && (kappa = q)
    end
    evaluated_omitted = kappa < evaluated_q ? maximum(@view ratios[(kappa + 2):end]) : 0.0
    diagnostics = (
        kappa=kappa,
        lmax=l,
        nmax=n_limit,
        nside=resolution,
        beta_max=beta,
        threshold=target,
        normalization=normalization,
        scope=:lmax_at_beta_max,
        q_values=collect(0:evaluated_q),
        ratios=ratios,
        minimum_diagonal=minimum_diagonal,
        tail_start=evaluated_q + 1,
        tail_bound=tail_bound,
        omitted_maximum_bound=max(evaluated_omitted, tail_bound),
    )
    return details ? diagnostics : kappa
end

function estimate_kappa(
    lmax::Integer,
    nmax::Integer,
    nside::Integer;
    threshold::Real=1e-2,
    kwargs...,
)
    return estimate_kappa(lmax, nmax, nside, threshold; kwargs...)
end
