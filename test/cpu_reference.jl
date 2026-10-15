# CPU reference implementations of each convolution stage and of one ring.
# Independent of the GPU kernels; used by test/runtests.jl and by the
# beam-batched pipeline test.

function _effective_beam_cpu_reference(
    table,
    moments,
    blm,
    l_first,
    l_last,
)
    T = typeof(table.beta_max)
    Mmax = maximum(abs, Int.(table.pair_M))
    M_count = 2 * Mmax + 1
    tile_count = l_last - l_first + 1
    pixel_count = size(moments, 5)
    output = zeros(
        Complex{T},
        3 * M_count * tile_count,
        blm.numofbeam,
        pixel_count,
        2,
    )
    pair_lookup = Dict(
        (Int(M), Int(n)) => pair_index
        for (pair_index, (M, n)) in enumerate(zip(table.pair_M, table.pair_n))
    )

    for (tile_index, l) in enumerate(l_first:l_last)
        l_index = l - table.lstart + 1
        nmax_l = min(table.nmax, l)
        blocks = blm_slice(blm, l)
        for M in -min(Mmax, l):min(Mmax, l)
            M_index = M + Mmax + 1
            for n in max(-nmax_l, M - table.kappa):min(nmax_l, M + table.kappa)
                pair_index = pair_lookup[(M, n)]
                delta_index = M - n + table.kappa + 1
                n_index = n + table.nmax + 1
                local_n_index = n + nmax_l + 1
                for pixel_index in 1:pixel_count,
                    scan_weight in 1:2
                    locald = sum(
                        table.coefficients[
                            beta_index,
                            pair_index,
                            l_index,
                        ] * moments[
                            scan_weight,
                            beta_index,
                            delta_index,
                            n_index,
                            pixel_index,
                        ]
                        for beta_index in axes(table.coefficients, 1)
                    )
                    for beam_index in 1:blm.numofbeam,
                        spin_index in 1:3
                        term_index = spin_index + 3 * (
                            M_index - 1 + M_count * (tile_index - 1)
                        )
                        output[
                            term_index,
                            beam_index,
                            pixel_index,
                            scan_weight,
                        ] += locald * blocks[spin_index][
                            local_n_index,
                            beam_index,
                        ]
                    end
                end
            end
        end
    end
    return output
end

function _sky_ring_cpu_reference(
    cp,
    ringd,
    alm,
    phi_pixels,
    l_first,
    l_last,
)
    T = typeof(real(zero(eltype(alm))))
    Mmax_global = calc_Mmax(cp, cp.lstop)
    M_count = 2 * Mmax_global + 1
    tile_count = l_last - l_first + 1
    output = zeros(
        Complex{T},
        alm.numofsky,
        3 * M_count * tile_count,
        length(phi_pixels),
    )
    for (tile_index, l) in enumerate(l_first:l_last)
        l_index = l - cp.lstart + 1
        Mmax_l = calc_Mmax(cp, l)
        blocks = alm_slice(alm, l)
        for (pixel_index, phi) in enumerate(phi_pixels),
            M in -Mmax_l:Mmax_l,
            spin_index in 1:3,
            sky_index in 1:alm.numofsky
            value = zero(Complex{T})
            for m in -l:l
                value +=
                    ringd[
                        m + l + 1,
                        M + Mmax_l + 1,
                        l_index,
                    ] * blocks[spin_index][
                        sky_index,
                        m + l + 1,
                    ] * cis(T(m) * T(phi))
            end
            spin_weight = spin_index == 1 ? one(T) : T(0.5)
            M_index = M + Mmax_global + 1
            term_index = spin_index + 3 * (
                M_index - 1 + M_count * (tile_index - 1)
            )
            output[sky_index, term_index, pixel_index] =
                spin_weight * value
        end
    end
    return output
end

function _convolution_tile_cpu_reference(
    accumulated,
    sky_ring,
    effective_beam,
)
    expected = copy(accumulated)
    sky_count, term_count, pixel_count = size(sky_ring)
    beam_count = size(effective_beam, 2)
    for scan_weight in 1:size(effective_beam, 4),
        pixel_index in 1:pixel_count,
        beam_index in 1:beam_count,
        sky_index in 1:sky_count
        expected[
            sky_index,
            beam_index,
            pixel_index,
            scan_weight,
        ] += sum(
            sky_ring[sky_index, term_index, pixel_index] *
            effective_beam[
                term_index,
                beam_index,
                pixel_index,
                scan_weight,
            ]
            for term_index in 1:term_count
        )
    end
    return expected
end

function _ring_convolution_cpu_reference(
    cp,
    geometry,
    positions,
    angles,
    table,
    prepared_alm,
    prepared_blm,
    ringd_host,
    plan,
    l_tile,
)
    T = typeof(plan.beta_max)
    moments_host = cat(
        (scan_moments_cpu(plan, angle) for angle in angles)...;
        dims=5,
    )
    expected = zeros(
        Complex{T},
        cp.numofsky,
        cp.numofbeam,
        length(positions),
        2,
    )
    for l_first in cp.lstart:l_tile:cp.lstop
        l_last = min(l_first + l_tile - 1, cp.lstop)
        expected = _convolution_tile_cpu_reference(
            expected,
            _sky_ring_cpu_reference(
                cp,
                ringd_host,
                prepared_alm,
                geometry.phi_pixels,
                l_first,
                l_last,
            )[:, :, positions],
            _effective_beam_cpu_reference(
                table,
                moments_host,
                prepared_blm,
                l_first,
                l_last,
            ),
        )
    end
    return expected
end
