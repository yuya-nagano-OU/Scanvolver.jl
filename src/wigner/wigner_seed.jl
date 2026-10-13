@inline _wignerd_rec_phase(k::Integer) = iseven(k) ? 1.0 : -1.0

struct _WignerdRecursionSeedPlan
    sign::Float64
    log_coefficient::Float64
    pow_c::Int
    pow_s::Int
end

function _precompute_wignerd_lmin_seed_plan(
    l::Integer,
    m::Integer,
    n::Integer,
)
    kmin = max(0, m - n)
    kmax = min(l + m, l - n)
    kmin == kmax || throw(ArgumentError(
        "seed is only valid at l = max(abs(m), abs(n))",
    ))

    k = kmin
    pow_c = 2l + m - n - 2k
    pow_s = n - m + 2k
    logpref = 0.5 * (
        loggamma(l + m + 1) +
        loggamma(l - m + 1) +
        loggamma(l + n + 1) +
        loggamma(l - n + 1)
    )
    logden =
        loggamma(l + m - k + 1) +
        loggamma(k + 1) +
        loggamma(n - m + k + 1) +
        loggamma(l - n - k + 1)

    return _WignerdRecursionSeedPlan(
        _wignerd_rec_phase(k),
        logpref - logden,
        pow_c,
        pow_s,
    )
end
