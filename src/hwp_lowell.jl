# Scalar effective-beam ell=0,1 coefficients from polarized input beam ell1>=2.
# The spatial Mueller mode is uniform in the beam-centred Ludwig-3 frame.
# For harmonic t=+/-2, its spherical-frame factor is exp(i*t*phi), so the
# polarized beam order is n-t. No spin-2 coefficient at ell<2 is introduced.

function _hwp_lowell_weights(ell::Int, n::Int, t::Int, lmax::Int)
    weights = zeros(Float64, lmax + 1)
    for ell1 in max(2, abs(n - t)):lmax
        # Cyclic permutations of the two Gaunt 3j symbols put ell2 first.
        # Fixing output ell=0,1 leaves only one or at most three ell2 values,
        # so these WignerFamilies calls and the whole sum are O(lmax).
        spin = wigner3j_f(ell, ell1, 0, -t)
        order = n == 0 ? spin : wigner3j_f(ell, ell1, -n, n - t)
        for ell2 in max(2, abs(ell1 - ell)):(ell1 + ell)
            phase = isodd(n + ell2) ? -1.0 : 1.0
            weights[ell1 + 1] += phase * 2(2ell2 + 1) / (ell2 * (ell2 + 1)) *
                sqrt(Float64((2ell1 + 1) * (2ell + 1))) * spin[ell2] * order[ell2]
        end
    end
    return weights
end

@inline function _hwp_lowell_source(blm, ell1, n, t, lmax, input_basis)
    k = n - t
    index = _packed_alm_index(ell1, abs(k), lmax)
    _, plus, minus = _positive_spin_coefficients(blm, index, input_basis, Float64)
    if k >= 0
        return conj(t == 2 ? plus : minus)
    end
    phase = isodd(k) ? -1.0 : 1.0
    return phase * (t == 2 ? minus : plus)
end

function _prepare_hwp_lowell(cp, blms, input_basis, C, ::Type{T}) where {T}
    result = Dict{Int,NTuple{2,Matrix{Complex{T}}}}()
    for ell in cp.lstart:min(1, cp.lstop)
        cosine = zeros(Complex{T}, 2ell + 1, cp.numofbeam)
        sine = similar(cosine)
        fill!(sine, 0)
        for (X, t) in ((2, 2), (3, -2))
            iszero(C[X, 1]) && continue
            factor = C[X, 1] / sqrt(2.0)
            for n in -ell:ell
                abs(n - t) <= cp.bnmax || continue
                weights = _hwp_lowell_weights(ell, n, t, cp.lmax)
                for beam in eachindex(blms)
                    value = zero(ComplexF64)
                    for ell1 in max(2, abs(n - t)):cp.lmax
                        value += weights[ell1 + 1] * _hwp_lowell_source(
                            blms[beam], ell1, n, t, cp.lmax, input_basis,
                        )
                    end
                    value *= factor
                    row = n + ell + 1
                    cosine[row, beam] += Complex{T}(value)
                    sine[row, beam] += Complex{T}((t > 0 ? im : -im) * value)
                end
            end
        end
        result[ell] = (cosine, sine)
    end
    return result
end
