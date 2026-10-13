function _check_scan_moment_angles(
    plan::ScanMomentPlan,
    angles::Angles,
)
    angles.num >= 0 || throw(ArgumentError(
        "angles.num must be non-negative",
    ))
    angles.num == length(angles.α) == length(angles.β) ==
        length(angles.γ) == length(angles.ψ) ||
        throw(DimensionMismatch(
            "Angles fields have inconsistent lengths",
        ))
    for beta in angles.β
        _scaled_beta(plan, beta)
    end
    return nothing
end

"""
    scan_moments_cpu(plan, angles) -> Array{Complex{T},4}

Compute the exact direct-sum scan moments for one pixel on the CPU. The
result has layout `(scan_weight, beta_mode, q, n)`, where the two scan
weights are `s=0` and `s=2`, `q=-plan.kappa:plan.kappa`, and
`n=-plan.nmax:plan.nmax`.
"""
function scan_moments_cpu(
    plan::ScanMomentPlan{T},
    angles::Angles,
) where {T<:_CHEBYSHEV_PRECISIONS}
    _check_scan_moment_angles(plan, angles)
    moments = zeros(
        Complex{T},
        2,
        plan.beta_order,
        2 * plan.kappa + 1,
        2 * plan.nmax + 1,
    )
    iszero(angles.num) && return moments

    chebyshev = Vector{T}(undef, plan.beta_order)
    inverse_hits = inv(T(angles.num))
    @inbounds for sample_index in 1:angles.num
        alpha = T(angles.α[sample_index])
        alpha_plus_gamma = alpha + T(angles.γ[sample_index])
        psi = T(angles.ψ[sample_index])
        _chebyshev_values!(
            chebyshev,
            plan,
            angles.β[sample_index],
        )

        for (n_index, n) in enumerate(-plan.nmax:plan.nmax)
            n_phase = T(n) * alpha_plus_gamma
            for (q_index, q) in enumerate(-plan.kappa:plan.kappa)
                spin0_phase = cis(T(q) * alpha + n_phase)
                spin2_phase = spin0_phase * cis(T(2) * psi)
                for beta_index in 1:plan.beta_order
                    weight = chebyshev[beta_index] * inverse_hits
                    moments[1, beta_index, q_index, n_index] +=
                        weight * spin0_phase
                    moments[2, beta_index, q_index, n_index] +=
                        weight * spin2_phase
                end
            end
        end
    end
    return moments
end
