using Scanvolver
using CUDA
using Healpix: Alm, almIndex, alm2map, numberOfAlms
using LinearAlgebra: cond
using Test

include("cpu_reference.jl")

# GPU conveniences for the tests. The HDF5 pipelines do the same inline.

"""ringD at one colatitude from a fresh context, as a device array the caller owns."""
function _ringd_gpu(cp, beta; precision=Float64)
    context = ringd_context(cp; precision=precision)
    try
        ringd = copy(ringd_blocks_recursion_gpu!(context, cp, beta))
        CUDA.synchronize()
        return ringd
    finally
        free_ringd_context!(context)
    end
end

"""One effective-beam tile: the beam-independent kernel, then every beam."""
function _effective_beam_gpu(table, moments, blm, l_first, l_last)
    kernel = build_effective_kernel_gpu(table, moments, l_first, l_last)
    try
        return apply_beams_gemm_gpu(table, kernel, blm, l_first, l_last)
    finally
        CUDA.synchronize()
        CUDA.unsafe_free!(kernel)
    end
end

"""
One ring through the stages the pipelines' `process_ring!` calls, in the same
order but without beam batching. The caller owns `result.accumulated`.
"""
function _convolve_ring_stages(
    cp, geometry, active, angles, alm, blm, table, plan;
    l_tile, moment_method=:gemm,
)
    T = typeof(plan.beta_max)
    context = ringd_context(cp; precision=T)
    plan_cache = Scanvolver._SkyRingFFTPlanCache()
    moments = nothing
    try
        ringd = ringd_blocks_recursion_gpu!(context, cp, geometry.theta)
        moments = scan_moments_gpu(plan, angles; method=moment_method)
        accumulated = CUDA.zeros(Complex{T}, cp.numofsky, cp.numofbeam, length(active), 2)
        for l_first in cp.lstart:l_tile:cp.lstop
            l_last = min(l_first + l_tile - 1, cp.lstop)
            kernel = build_effective_kernel_gpu(table, moments, l_first, l_last)
            sky = build_sky_ring_gpu(
                cp, ringd, alm, geometry.phi_pixels, l_first, l_last; plan_cache=plan_cache,
            )
            active_sky = select_active_sky_ring_gpu(sky, active)
            effective = apply_beams_gemm_gpu(table, kernel, blm, l_first, l_last)
            accumulate_convolution_tile_gemm_gpu!(accumulated, active_sky, effective)
            CUDA.synchronize()
            free_active_sky_ring!(active_sky, sky)
            foreach(CUDA.unsafe_free!, (effective, sky, kernel))
        end
        positions = Int.(Array(active.positions))
        return RingConvolution{T}(
            geometry.ring, geometry.first_pixel, geometry.pixel_count,
            positions, geometry.first_pixel .+ positions .- 1, accumulated,
        )
    finally
        CUDA.synchronize()
        Scanvolver._release_sky_ring_fft_plans!(plan_cache)
        moments === nothing || CUDA.unsafe_free!(moments)
        free_ringd_context!(context)
    end
end

@testset "ConvolutionParams" begin
    cp = ConvolutionParams(nside_output=1)
    @test cp.nside_output == 1
    @test cp.lstart == 0
    @test cp.lstop == 2
    @test cp.lmax == 2
    @test cp.Mmax_max == 2
    @test cp.noise == false
    @test cp.noise_sigma === nothing
    @test cp.HWP == false
    @test cp.HWP_rpm === nothing
    @test cp.resol.nside == 1
    @test !ismutabletype(typeof(cp))
    @test occursin("ConvolutionParams", sprint(show, cp))

    original = ConvolutionParams(
        nside_output=16,
        lstop=47,
        lmax=47,
        bnmax=2,
        κ=20,
    )
    updated = ConvolutionParams(original; κ=10)
    @test original.κ == 20
    @test original.Mmax_max == 22
    @test updated.κ == 10
    @test updated.Mmax_max == 12
    @test updated.resol.nside == original.resol.nside
    @test_throws ErrorException setproperty!(original, :κ, 10)

    original = ConvolutionParams(original; nside_output=32)
    @test original.nside_output == 32
    @test original.resol.nside == 32
    @test original.lstop == 47
    @test original.lmax == 47

    cp_with_sigma =
        ConvolutionParams(nside_output=1, noise_sigma=0.1)
    @test cp_with_sigma.noise == true
    @test cp_with_sigma.noise_sigma == 0.1

    cp_with_hwp =
        ConvolutionParams(nside_output=1, HWP=true, HWP_rpm=46)
    @test cp_with_hwp.HWP == true
    @test cp_with_hwp.HWP_rpm == 46.0

    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        lstart=2,
        lstop=1,
    )
    @test_throws ArgumentError ConvolutionParams(nside_output=0)
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        lstart=-1,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        lstop=3,
        lmax=2,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        numofsky=0,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        numofbeam=0,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        lmax=1,
        bnmax=2,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        bnmax=-1,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        κ=-1,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        noise=true,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        noise_sigma=-0.1,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        noise_sigma=Inf,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        HWP=true,
    )
    @test_throws ArgumentError ConvolutionParams(
        nside_output=1,
        HWP=true,
        HWP_rpm=Inf,
    )
end

@testset "prepare_alm" begin
    cp = ConvolutionParams(
        nside_output=2,
        lstart=0,
        lstop=3,
        lmax=3,
        numofsky=1,
    )
    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    teb = zeros(ComplexF64, 3, coefficient_count)

    l = 2
    m = 1
    packed_index = m * (cp.lmax + 1) - m * (m - 1) ÷ 2 + (l - m) + 1
    teb[:, packed_index] .= ComplexF64[1 + 2im, 3 - 0.5im, -2 + 4im]

    prepared = prepare_alm(cp, teb)
    @test prepared isa PreparedAlm{Float64}
    @test !ismutabletype(typeof(prepared))
    @test length(prepared) == 4
    @test prepared.lstart == 0
    @test prepared.lstop == 3
    @test prepared.lmax == 3
    @test prepared.numofsky == 1
    @test eltype(prepared) === ComplexF64
    @test occursin("PreparedAlm{Float64}", sprint(show, prepared))

    spin0, spin_plus2, spin_minus2 = alm_slice(prepared, l)
    @test size(spin0) == (1, 2l + 1)
    @test size(spin_plus2) == size(spin0)
    @test size(spin_minus2) == size(spin0)
    positive_index = m + l + 1
    negative_index = -m + l + 1
    expected_spin0 = teb[1, packed_index]
    expected_plus2 = -(teb[2, packed_index] + im * teb[3, packed_index])
    expected_minus2 = -(teb[2, packed_index] - im * teb[3, packed_index])
    @test spin0[1, positive_index] == expected_spin0
    @test spin_plus2[1, positive_index] == expected_plus2
    @test spin_minus2[1, positive_index] == expected_minus2
    @test spin0[1, negative_index] == -conj(expected_spin0)
    @test spin_plus2[1, negative_index] == -conj(expected_minus2)
    @test spin_minus2[1, negative_index] == -conj(expected_plus2)

    spin_input = similar(teb)
    spin_input[1, :] .= teb[1, :]
    spin_input[2, :] .= -(teb[2, :] .+ im .* teb[3, :])
    spin_input[3, :] .= -(teb[2, :] .- im .* teb[3, :])
    prepared_spin = prepare_alm(cp, spin_input; input_basis=:spin)
    for l_test in cp.lstart:cp.lstop
        teb_blocks = alm_slice(prepared, l_test)
        spin_blocks = alm_slice(prepared_spin, l_test)
        @test all(teb_blocks[index] == spin_blocks[index] for index in 1:3)
    end

    cp_multi = ConvolutionParams(
        nside_output=2,
        lstart=1,
        lstop=3,
        lmax=3,
        numofsky=2,
    )
    prepared_multi = prepare_alm(
        cp_multi,
        [teb, 2 .* teb];
        precision=Float32,
    )
    @test prepared_multi isa PreparedAlm{Float32}
    @test prepared_multi.numofsky == 2
    @test size(alm_slice(prepared_multi, 2)[1]) == (2, 5)
    @test alm_slice(prepared_multi, 2)[1][2, :] ≈
        2 .* alm_slice(prepared_multi, 2)[1][1, :]

    @test_throws ArgumentError alm_slice(prepared, -1)
    @test_throws ArgumentError alm_slice(prepared, 4)
    @test_throws ArgumentError prepare_alm(
        cp,
        teb;
        input_basis=:unknown,
    )
    @test_throws ArgumentError prepare_alm(cp, teb; precision=Float16)
    @test_throws DimensionMismatch prepare_alm(cp, teb[:, 1:end-1])
    @test_throws DimensionMismatch prepare_alm(cp, zeros(2, coefficient_count))
    @test_throws ArgumentError prepare_alm(cp_multi, [teb])
    @test_throws ArgumentError prepare_alm(cp, Any[1])
    @test_throws ArgumentError prepare_alm(cp, Matrix[])
end

@testset "prepare_blm" begin
    cp = ConvolutionParams(
        nside_output=2,
        lstart=0,
        lstop=4,
        lmax=4,
        bnmax=2,
        numofbeam=1,
    )
    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    teb = zeros(ComplexF64, 3, coefficient_count)

    l = 3
    for n in 1:2
        packed_index =
            n * (cp.lmax + 1) - n * (n - 1) ÷ 2 + (l - n) + 1
        teb[:, packed_index] .= ComplexF64[
            n + (n + 1)im,
            2n - 0.25im,
            -n + (3n - 1)im,
        ]
    end

    prepared = prepare_blm(cp, teb)
    @test prepared isa PreparedBlm{Float64}
    @test !ismutabletype(typeof(prepared))
    @test length(prepared) == 5
    @test prepared.lstart == 0
    @test prepared.lstop == 4
    @test prepared.lmax == 4
    @test prepared.bnmax == 2
    @test prepared.numofbeam == 1
    @test eltype(prepared) === ComplexF64
    @test occursin("PreparedBlm{Float64}", sprint(show, prepared))

    for l_test in cp.lstart:cp.lstop
        nmax = calc_nmax(cp, l_test)
        blocks = blm_slice(prepared, l_test)
        @test all(size(block) == (2nmax + 1, 1) for block in blocks)
    end

    blocks = blm_slice(prepared, l)
    for n in 1:2
        packed_index =
            n * (cp.lmax + 1) - n * (n - 1) ÷ 2 + (l - n) + 1
        raw_spin0 = teb[1, packed_index]
        raw_plus2 = -(teb[2, packed_index] + im * teb[3, packed_index])
        raw_minus2 = -(teb[2, packed_index] - im * teb[3, packed_index])
        phase = isodd(n) ? -1 : 1
        positive_index = n + calc_nmax(cp, l) + 1
        negative_index = -n + calc_nmax(cp, l) + 1

        @test blocks[1][positive_index, 1] == conj(raw_spin0)
        @test blocks[2][positive_index, 1] == conj(raw_plus2)
        @test blocks[3][positive_index, 1] == conj(raw_minus2)
        @test blocks[1][negative_index, 1] == phase * raw_spin0
        @test blocks[2][negative_index, 1] == phase * raw_minus2
        @test blocks[3][negative_index, 1] == phase * raw_plus2

        # Check the symmetry again using only prepared coefficients. This
        # independently verifies the spin swap under n -> -n.
        @test blocks[1][negative_index, 1] ==
            phase * conj(blocks[1][positive_index, 1])
        @test blocks[2][negative_index, 1] ==
            phase * conj(blocks[3][positive_index, 1])
        @test blocks[3][negative_index, 1] ==
            phase * conj(blocks[2][positive_index, 1])
    end

    spin_input = similar(teb)
    spin_input[1, :] .= teb[1, :]
    spin_input[2, :] .= -(teb[2, :] .+ im .* teb[3, :])
    spin_input[3, :] .= -(teb[2, :] .- im .* teb[3, :])
    prepared_spin = prepare_blm(cp, spin_input; input_basis=:spin)
    for l_test in cp.lstart:cp.lstop
        teb_blocks = blm_slice(prepared, l_test)
        spin_blocks = blm_slice(prepared_spin, l_test)
        @test all(teb_blocks[index] == spin_blocks[index] for index in 1:3)
    end

    cp_multi = ConvolutionParams(cp; numofbeam=2)
    prepared_multi = prepare_blm(
        cp_multi,
        [teb, 2 .* teb];
        precision=Float32,
    )
    @test prepared_multi isa PreparedBlm{Float32}
    @test prepared_multi.numofbeam == 2
    @test size(blm_slice(prepared_multi, 3)[1]) == (5, 2)
    @test blm_slice(prepared_multi, 3)[1][:, 2] ≈
        2 .* blm_slice(prepared_multi, 3)[1][:, 1]

    @test_throws ArgumentError blm_slice(prepared, -1)
    @test_throws ArgumentError blm_slice(prepared, 5)
    @test_throws ArgumentError prepare_blm(
        cp,
        teb;
        input_basis=:unknown,
    )
    @test_throws ArgumentError prepare_blm(cp, teb; precision=Float16)
    @test_throws DimensionMismatch prepare_blm(cp, teb[:, 1:end-1])
    @test_throws DimensionMismatch prepare_blm(cp, zeros(2, coefficient_count))
    @test_throws ArgumentError prepare_blm(cp_multi, [teb])
    @test_throws ArgumentError prepare_blm(cp, Any[1])
    @test_throws ArgumentError prepare_blm(cp, Matrix[])
    hwp_cp = ConvolutionParams(cp; HWP=true, HWP_rpm=60)
    @test_throws ArgumentError prepare_blm(hwp_cp, teb)
end

@testset "Wigner mode bounds" begin
    cp = ConvolutionParams(
        nside_output=8,
        lstop=12,
        lmax=12,
        bnmax=3,
        κ=4,
    )
    @test calc_Mmax(cp, 0) == 0
    @test calc_Mmax(cp, 5) == 5
    @test calc_Mmax(cp, 12) == 7
    @test calc_nmax(cp, 0) == 0
    @test calc_nmax(cp, 2) == 2
    @test calc_nmax(cp, 12) == 3
end

@testset "ringD seed plan" begin
    seed = Scanvolver._precompute_wignerd_lmin_seed_plan(
        0,
        0,
        0,
    )
    @test seed.sign == 1.0
    @test seed.log_coefficient == 0.0
    @test seed.pow_c == 0
    @test seed.pow_s == 0

    @test_throws ArgumentError Scanvolver._precompute_wignerd_lmin_seed_plan(2, 0, 0)
end

@testset "Chebyshev" begin
    plan = ScanMomentPlan(
        kappa=3,
        nmax=2,
        beta_order=4,
        beta_max=2.0,
        precision=Float64,
    )
    @test plan.kappa == 3
    @test plan.nmax == 2
    @test plan.beta_order == 4
    @test plan.beta_max === 2.0

    nodes = chebyshev_nodes(plan)
    expected_nodes = [
        1 + cos(pi / 8),
        1 + cos(3pi / 8),
        1 + cos(5pi / 8),
        1 + cos(7pi / 8),
    ]
    @test nodes ≈ expected_nodes
    @test all(0 .< nodes .< plan.beta_max)

    values = zeros(Float64, plan.beta_order)
    returned = Scanvolver._chebyshev_values!(values, plan, 1.2)
    x = 0.2
    @test returned === values
    @test values ≈ [1.0, x, 2x^2 - 1, 4x^3 - 3x]
    @test Scanvolver._scaled_beta(plan, 0.0) == -1.0
    @test Scanvolver._scaled_beta(plan, plan.beta_max) == 1.0
    @test_throws DomainError Scanvolver._scaled_beta(plan, -0.1)
    @test_throws DomainError Scanvolver._scaled_beta(plan, 2.1)
    @test_throws DimensionMismatch Scanvolver._chebyshev_values!(
        zeros(Float64, 3),
        plan,
        1.0,
    )

    transform = Scanvolver._chebyshev_transform(plan)
    coefficients = [3.0, 2.0, -0.5, 0.25]
    node_values = similar(nodes)
    workspace = similar(nodes)
    for node_index in eachindex(nodes)
        Scanvolver._chebyshev_values!(
            workspace,
            plan,
            nodes[node_index],
        )
        node_values[node_index] = sum(coefficients .* workspace)
    end
    @test transform * node_values ≈ coefficients atol=2e-14 rtol=2e-14

    single = ScanMomentPlan(
        kappa=0,
        nmax=0,
        beta_order=1,
        beta_max=0.5,
    )
    single_values = zeros(Float32, 1)
    @test Scanvolver._chebyshev_values!(
        single_values,
        single,
        0.25,
    ) == Float32[1]
    @test eltype(chebyshev_nodes(single)) === Float32

    @test_throws ArgumentError ScanMomentPlan(
        kappa=-1,
        nmax=0,
        beta_order=4,
        beta_max=1.0,
    )
    @test_throws ArgumentError ScanMomentPlan(
        kappa=0,
        nmax=-1,
        beta_order=4,
        beta_max=1.0,
    )
    @test_throws ArgumentError ScanMomentPlan(
        kappa=0,
        nmax=0,
        beta_order=0,
        beta_max=1.0,
    )
    @test_throws ArgumentError ScanMomentPlan(
        kappa=0,
        nmax=0,
        beta_order=4,
        beta_max=0.0,
    )
    @test_throws ArgumentError ScanMomentPlan(
        kappa=0,
        nmax=0,
        beta_order=4,
        beta_max=Inf,
    )
    @test_throws ArgumentError ScanMomentPlan(
        kappa=0,
        nmax=0,
        beta_order=4,
        beta_max=1.0,
        precision=Float16,
    )
end

function _synthetic_scan_moment_angles(
    cp;
    pixel::Int=1,
    hit_count::Int=11,
)
    theta_pixel, phi_pixel = Scanvolver.pix2angRing(
        cp.resol,
        pixel,
    )
    phases = range(0.0, 2pi; length=hit_count + 1)[1:end-1]
    theta = Float64[
        theta_pixel + 2.5e-4 * sin(phase)
        for phase in phases
    ]
    phi = Float64[
        phi_pixel + 2.0e-4 * cos(phase)
        for phase in phases
    ]
    psi = Float64[
        mod(0.31 + 1.7 * phase, 2pi)
        for phase in phases
    ]
    return Angles(
        cp;
        pix=pixel,
        φ=phi,
        θ=theta,
        ψ=psi,
    )
end

@testset "Scan moments CPU" begin
    cp = ConvolutionParams(
        nside_output=4,
        lstop=8,
        lmax=8,
        bnmax=1,
        κ=2,
    )
    angles = _synthetic_scan_moment_angles(cp)
    plan = ScanMomentPlan(
        kappa=cp.κ,
        nmax=calc_nmax(cp, cp.lstop),
        beta_order=6,
        beta_max=1.05 * maximum(angles.β),
        precision=Float64,
    )
    moments = scan_moments_cpu(plan, angles)

    @test size(moments) == (2, 6, 5, 3)
    @test eltype(moments) === ComplexF64
    @test all(isfinite, moments)
    q_zero = plan.kappa + 1
    n_zero = plan.nmax + 1
    @test moments[1, 1, q_zero, n_zero] ≈
        1.0 atol=2e-15 rtol=2e-15
    expected_spin2 = sum(cis.(2 .* angles.ψ)) / angles.num
    @test moments[2, 1, q_zero, n_zero] ≈
        expected_spin2 atol=2e-15 rtol=2e-15

    plan32 = ScanMomentPlan(
        kappa=cp.κ,
        nmax=calc_nmax(cp, cp.lstop),
        beta_order=6,
        beta_max=plan.beta_max,
        precision=Float32,
    )
    @test eltype(scan_moments_cpu(plan32, angles)) === ComplexF32

    inconsistent = deepcopy(angles)
    inconsistent.num += 1
    @test_throws DimensionMismatch scan_moments_cpu(
        plan,
        inconsistent,
    )

    outside_interval = deepcopy(angles)
    outside_interval.β[1] = 2 * plan.beta_max
    @test_throws DomainError scan_moments_cpu(
        plan,
        outside_interval,
    )
end

@testset "Scan moments GPU" begin
    cp = ConvolutionParams(
        nside_output=4,
        lstop=8,
        lmax=8,
        bnmax=1,
        κ=2,
    )
    angles = [
        _synthetic_scan_moment_angles(
            cp;
            pixel=pixel,
            hit_count=8 + pixel,
        )
        for pixel in 1:3
    ]
    beta_max = 1.05 * maximum(
        maximum(angle.β)
        for angle in angles
    )

    if CUDA.functional()
        for precision in (Float32, Float64)
            plan = ScanMomentPlan(
                kappa=cp.κ,
                nmax=calc_nmax(cp, cp.lstop),
                beta_order=6,
                beta_max=beta_max,
                precision=precision,
            )
            expected = [
                scan_moments_cpu(plan, angle)
                for angle in angles
            ]
            direct_device = CUDA.@sync scan_moments_direct_gpu(
                plan,
                angles,
            )
            gemm_device = CUDA.@sync scan_moments_gemm_gpu(
                plan,
                angles,
            )
            default_device = CUDA.@sync scan_moments_gpu(
                plan,
                angles,
            )
            direct = Array(direct_device)
            gemm = Array(gemm_device)
            default = Array(default_device)
            CUDA.unsafe_free!(direct_device)
            CUDA.unsafe_free!(gemm_device)
            CUDA.unsafe_free!(default_device)

            @test size(gemm) == (2, 6, 5, 3, 3)
            @test eltype(gemm) === Complex{precision}
            tolerance = precision == Float32 ? 2e-5 : 2e-12
            @test default ≈ gemm atol=tolerance rtol=tolerance
            for pixel_index in eachindex(angles)
                @test @view(direct[:, :, :, :, pixel_index]) ≈
                    expected[pixel_index] atol=tolerance rtol=tolerance
                @test @view(gemm[:, :, :, :, pixel_index]) ≈
                    expected[pixel_index] atol=tolerance rtol=tolerance
            end
        end

        plan = ScanMomentPlan(
            kappa=cp.κ,
            nmax=calc_nmax(cp, cp.lstop),
            beta_order=6,
            beta_max=beta_max,
        )
        @test_throws ArgumentError scan_moments_gpu(plan, Angles[])
        @test_throws ArgumentError scan_moments_gemm_gpu(
            plan,
            Angles[],
        )
        @test scan_moments_gemm_memory_bytes(plan, angles) > 0

        direct_device = CUDA.@sync scan_moments_direct_gpu(plan, angles)
        fallback_device = @test_logs (
            :warn,
            r"memory budget",
        ) begin
            CUDA.@sync scan_moments_gpu(
                plan,
                angles;
                memory_fraction=0.0,
            )
        end
        @test Array(fallback_device) == Array(direct_device)
        CUDA.unsafe_free!(direct_device)
        CUDA.unsafe_free!(fallback_device)
        @test_throws ArgumentError scan_moments_gpu(
            plan,
            angles;
            memory_fraction=0.0,
            fallback_to_direct=false,
        )
        @test_throws ArgumentError scan_moments_gpu(
            plan,
            angles;
            method=:unknown,
        )
        @test_throws ArgumentError scan_moments_gpu(
            plan,
            angles;
            memory_fraction=1.1,
        )
    else
        plan = ScanMomentPlan(
            kappa=cp.κ,
            nmax=calc_nmax(cp, cp.lstop),
            beta_order=6,
            beta_max=beta_max,
        )
        @test_throws ArgumentError scan_moments_gpu(plan, angles)
        @test_throws ArgumentError scan_moments_gemm_gpu(
            plan,
            angles,
        )
        @test_throws ArgumentError scan_moments_direct_gpu(
            plan,
            angles,
        )
    end
end

@testset "Effective beam GPU" begin
    cp = ConvolutionParams(
        nside_output=2,
        lstart=0,
        lstop=5,
        lmax=5,
        bnmax=2,
        κ=2,
        numofbeam=2,
    )
    angles = [
        _synthetic_scan_moment_angles(
            cp;
            pixel=pixel,
            hit_count=8 + pixel,
        )
        for pixel in 1:3
    ]
    beta_max = 1.05 * maximum(
        maximum(angle.β)
        for angle in angles
    )
    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    raw_blms = [
        ComplexF64[
            complex(
                sin(0.3 * spin + 0.2 * coefficient + beam),
                cos(0.4 * spin - 0.1 * coefficient + 0.5 * beam),
            )
            for spin in 1:3, coefficient in 1:coefficient_count
        ]
        for beam in 1:cp.numofbeam
    ]

    if CUDA.functional()
        for precision in (Float32, Float64)
            plan = ScanMomentPlan(
                kappa=cp.κ,
                nmax=calc_nmax(cp, cp.lstop),
                beta_order=10,
                beta_max=beta_max,
                precision=precision,
            )
            table = build_wigner_chebyshev_table_cpu(cp, plan)
            device_table = upload_wigner_table_gpu(table)
            moments_device = CUDA.@sync scan_moments_gemm_gpu(
                plan,
                angles,
            )
            moments_host = cat(
                (scan_moments_cpu(plan, angle) for angle in angles)...;
                dims=5,
            )
            prepared_blm = prepare_blm(
                cp,
                raw_blms;
                precision=precision,
            )
            l_first = 1
            l_last = 5
            effective_kernel = nothing
            effective = nothing
            try
                effective_kernel = CUDA.@sync build_effective_kernel_gpu(
                    device_table,
                    moments_device,
                    l_first,
                    l_last,
                )
                Mmax = Int(device_table.Mmax)
                @test size(effective_kernel) == (
                    2 * cp.κ + 1,
                    length(angles),
                    (2 * Mmax + 1) * (l_last - l_first + 1),
                    2,
                )

                effective = CUDA.@sync apply_beams_gemm_gpu(
                    device_table,
                    effective_kernel,
                    prepared_blm,
                    l_first,
                    l_last,
                )
                expected = _effective_beam_cpu_reference(
                    table,
                    moments_host,
                    prepared_blm,
                    l_first,
                    l_last,
                )
                tolerance = precision == Float32 ? 2e-4 : 2e-10
                @test Array(effective) ≈
                    expected atol=tolerance rtol=tolerance

                reused = CUDA.@sync apply_beams_gemm_gpu(
                    device_table,
                    effective_kernel,
                    prepared_blm,
                    l_first,
                    l_last,
                )
                @test Array(reused) == Array(effective)
                CUDA.unsafe_free!(reused)

                @test_throws ArgumentError build_effective_kernel_gpu(
                    device_table,
                    moments_device,
                    -1,
                    2,
                )
                @test_throws ArgumentError apply_beams_gemm_gpu(
                    device_table,
                    effective_kernel,
                    prepared_blm,
                    3,
                    2,
                )
            finally
                effective_kernel === nothing ||
                    CUDA.unsafe_free!(effective_kernel)
                effective === nothing || CUDA.unsafe_free!(effective)
                CUDA.unsafe_free!(moments_device)
                CUDA.unsafe_free!(device_table.coefficients)
                CUDA.unsafe_free!(device_table.pair_lookup)
            end
        end
    end
end

@testset "Sky ring GPU" begin
    cp = ConvolutionParams(
        nside_output=2,
        lstart=0,
        lstop=5,
        lmax=5,
        bnmax=2,
        κ=2,
        numofsky=2,
    )
    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    raw_alms = [
        ComplexF64[
            complex(
                sin(0.2 * spin + 0.3 * coefficient + sky),
                cos(0.4 * spin - 0.1 * coefficient + 0.5 * sky),
            )
            for spin in 1:3, coefficient in 1:coefficient_count
        ]
        for sky in 1:cp.numofsky
    ]
    pixel_count = 5
    phi0 = 0.17
    phi_pixels = [
        phi0 + 2pi * (pixel_index - 1) / pixel_count
        for pixel_index in 1:pixel_count
    ]

    if CUDA.functional()
        for precision in (Float32, Float64)
            prepared = prepare_alm(
                cp,
                raw_alms;
                precision=precision,
            )
            alm_gpu = upload_alm_gpu(prepared)
            @test alm_gpu isa GPUAlm{precision}
            @test alm_gpu.lstart == cp.lstart
            @test alm_gpu.lstop == cp.lstop
            @test alm_gpu.lmax == cp.lmax
            @test alm_gpu.numofsky == cp.numofsky
            @test eltype(alm_gpu) === Complex{precision}
            @test size(alm_gpu.values) == (
                2 * cp.lstop + 1,
                3,
                cp.numofsky,
                cp.lstop - cp.lstart + 1,
            )
            @test occursin("GPUAlm{$precision}", sprint(show, alm_gpu))

            packed_host = Array(alm_gpu.values)
            for l in cp.lstart:cp.lstop, spin_index in 1:3
                l_index = l - cp.lstart + 1
                m_count = 2 * l + 1
                @test packed_host[
                    1:m_count,
                    spin_index,
                    :,
                    l_index,
                ] == permutedims(
                    alm_slice(prepared, l)[spin_index],
                    (2, 1),
                )
                @test all(iszero, packed_host[
                    (m_count + 1):size(packed_host, 1),
                    spin_index,
                    :,
                    l_index,
                ])
            end

            ringd = CUDA.@sync _ringd_gpu(
                cp,
                0.73;
                precision=precision,
            )
            l_first = 1
            l_last = 5
            sky_ring = nothing
            repeated = nothing
            bad_ringd = nothing
            try
                empty!(CUDA.CUFFT.idle_handles)
                sky_ring = CUDA.@sync build_sky_ring_gpu(
                    cp,
                    ringd,
                    alm_gpu,
                    phi_pixels,
                    l_first,
                    l_last,
                )
                @test isempty(CUDA.CUFFT.idle_handles.idle_handles)
                @test isempty(CUDA.CUFFT.idle_handles.active_handles)
                M_count = 2 * calc_Mmax(cp, cp.lstop) + 1
                @test size(sky_ring) == (
                    cp.numofsky,
                    3 * M_count * (l_last - l_first + 1),
                    pixel_count,
                )
                expected = _sky_ring_cpu_reference(
                    cp,
                    Array(ringd),
                    prepared,
                    phi_pixels,
                    l_first,
                    l_last,
                )
                tolerance = precision == Float32 ? 5e-5 : 2e-11
                @test Array(sky_ring) ≈
                    expected atol=tolerance rtol=tolerance

                repeated = CUDA.@sync build_sky_ring_gpu(
                    cp,
                    ringd,
                    alm_gpu,
                    phi_pixels,
                    l_first,
                    l_last,
                )
                @test isempty(CUDA.CUFFT.idle_handles.idle_handles)
                @test isempty(CUDA.CUFFT.idle_handles.active_handles)
                @test Array(repeated) ≈
                    expected atol=tolerance rtol=tolerance

                bad_phi = copy(phi_pixels)
                bad_phi[3] += 0.01
                @test_throws ArgumentError build_sky_ring_gpu(
                    cp,
                    ringd,
                    alm_gpu,
                    bad_phi,
                    l_first,
                    l_last,
                )
                @test_throws ArgumentError build_sky_ring_gpu(
                    cp,
                    ringd,
                    alm_gpu,
                    phi_pixels,
                    -1,
                    l_last,
                )
                bad_ringd = CUDA.zeros(
                    Complex{precision},
                    1,
                    1,
                    1,
                )
                @test_throws DimensionMismatch build_sky_ring_gpu(
                    cp,
                    bad_ringd,
                    alm_gpu,
                    phi_pixels,
                    l_first,
                    l_last,
                )
            finally
                sky_ring === nothing || CUDA.unsafe_free!(sky_ring)
                repeated === nothing || CUDA.unsafe_free!(repeated)
                bad_ringd === nothing || CUDA.unsafe_free!(bad_ringd)
                CUDA.unsafe_free!(ringd)
                CUDA.unsafe_free!(alm_gpu.values)
            end
        end
    else
        prepared = prepare_alm(cp, raw_alms)
        @test_throws ArgumentError upload_alm_gpu(prepared)
    end
end

function _synthetic_convolution_tile_inputs(
    ::Type{T},
    sky_count,
    term_count,
    pixel_count,
    beam_count,
    seed,
) where {T}
    sky_ring = Complex{T}[
        complex(
            T(sin(0.31 * sky + 0.17 * term + 0.11 * pixel + seed)),
            T(cos(0.23 * sky - 0.29 * term + 0.37 * pixel - seed)),
        )
        for sky in 1:sky_count,
            term in 1:term_count,
            pixel in 1:pixel_count
    ]
    effective_beam = Complex{T}[
        complex(
            T(cos(0.19 * term + 0.41 * beam + 0.13 * pixel - seed)),
            T(sin(0.43 * term - 0.07 * beam + 0.29 * pixel + seed)),
        )
        for term in 1:term_count,
            beam in 1:beam_count,
            pixel in 1:pixel_count,
            scan_weight in 1:2
    ]
    return sky_ring, effective_beam
end

@testset "Ring geometry" begin
    cp = ConvolutionParams(nside_output=4, lstop=8, lmax=8)

    @test ring_count(cp) == 4 * 4 - 1
    @test ring_pixel_range(cp, 1) == (1, 4)
    @test ring_pixel_range(cp, 2) == (5, 12)
    @test ring_pixel_range(cp, 4) == (25, 40)
    @test ring_pixel_range(cp, 5) == (41, 56)
    @test ring_pixel_range(cp, 8) == (89, 104)
    @test ring_pixel_range(cp, 15) == (189, 192)
    @test_throws ArgumentError ring_pixel_range(cp, 0)
    @test_throws ArgumentError ring_geometry(cp, ring_count(cp) + 1)

    for ring in (1, 2, 4, 5, 8, 12, 15)
        geometry = ring_geometry(cp, ring)
        first_pixel, last_pixel = ring_pixel_range(cp, ring)
        @test geometry.ring == ring
        @test geometry.first_pixel == first_pixel
        @test geometry.pixel_count == last_pixel - first_pixel + 1
        @test length(geometry) == geometry.pixel_count
        @test length(geometry.phi_pixels) == geometry.pixel_count

        # One colatitude per ring: a single ringD stack serves the ring.
        for pixel in first_pixel:last_pixel
            theta, phi = Scanvolver.pix2angRing(cp.resol, pixel)
            @test theta ≈ geometry.theta atol=1e-12
            @test geometry.phi_pixels[pixel - first_pixel + 1] ≈
                phi atol=1e-12
        end

        # Ascending, and accepted by the sky-ring FFT validator.
        @test issorted(geometry.phi_pixels)
        @test Scanvolver._check_ring_longitudes(geometry.phi_pixels) ≈
            geometry.phi_pixels[1]
    end
    @test occursin("ring=8", sprint(show, ring_geometry(cp, 8)))

    # Scan binning.
    ring = 5
    geometry = ring_geometry(cp, ring)
    hit_positions = [2, 7, 11]
    hits_per_pixel = 4
    phi_tod = Float64[]
    theta_tod = Float64[]
    psi_tod = Float64[]
    time_tod = Float64[]
    for position in hit_positions, hit in 1:hits_per_pixel
        pixel = geometry.first_pixel + position - 1
        theta_pixel, phi_pixel = Scanvolver.pix2angRing(cp.resol, pixel)
        push!(theta_tod, theta_pixel + 1e-4 * sin(hit))
        push!(phi_tod, phi_pixel + 1e-4 * cos(hit))
        push!(psi_tod, mod(0.3 + 0.7 * hit + position, 2pi))
        push!(time_tod, 0.1 * length(psi_tod))
    end
    # Samples belonging to a different ring must be ignored.
    other = ring_geometry(cp, ring + 2)
    other_theta, other_phi = Scanvolver.pix2angRing(
        cp.resol,
        other.first_pixel,
    )
    push!(theta_tod, other_theta)
    push!(phi_tod, other_phi)
    push!(psi_tod, 0.4)
    push!(time_tod, 99.0)

    if CUDA.functional()
        active, angles = make_ring_angles(
            cp,
            geometry,
            phi_tod,
            theta_tod,
            psi_tod,
        )
        try
            @test active isa ActiveRingPixels
            @test Array(active.positions) == Int32.(hit_positions)
            @test active.ring_pixel_count == geometry.pixel_count
            @test active.complete == false
            @test length(angles) == length(hit_positions)

            # angles[k] must be the pixel at positions[k]: this ordering is
            # what aligns the gathered sky ring with the effective beam.
            for (active_index, position) in pairs(hit_positions)
                pixel = geometry.first_pixel + position - 1
                theta_pixel, phi_pixel = Scanvolver.pix2angRing(
                    cp.resol,
                    pixel,
                )
                @test angles[active_index].θ_pix ≈ theta_pixel
                @test angles[active_index].φ_pix ≈ phi_pixel
                @test angles[active_index].num == hits_per_pixel
                @test angles[active_index].θ_pix ≈ geometry.theta
            end
            @test sum(angle.num for angle in angles) ==
                length(hit_positions) * hits_per_pixel
        finally
            CUDA.unsafe_free!(active.positions)
        end

        # A ring with no hits.
        empty_active, empty_angles = make_ring_angles(
            cp,
            ring_geometry(cp, 1),
            phi_tod,
            theta_tod,
            psi_tod,
        )
        @test empty_active === nothing
        @test isempty(empty_angles)

        # A fully hit ring reports complete and needs no gather.
        full_geometry = ring_geometry(cp, 1)
        full_phi = Float64[]
        full_theta = Float64[]
        full_psi = Float64[]
        for position in 1:full_geometry.pixel_count
            pixel = full_geometry.first_pixel + position - 1
            theta_pixel, phi_pixel = Scanvolver.pix2angRing(
                cp.resol,
                pixel,
            )
            push!(full_theta, theta_pixel)
            push!(full_phi, phi_pixel)
            push!(full_psi, 0.11 * position)
        end
        full_active, full_angles = make_ring_angles(
            cp,
            full_geometry,
            full_phi,
            full_theta,
            full_psi,
        )
        try
            @test full_active.complete == true
            @test length(full_angles) == full_geometry.pixel_count
        finally
            CUDA.unsafe_free!(full_active.positions)
        end
    end

    @test_throws DimensionMismatch make_ring_angles(
        cp,
        geometry,
        phi_tod,
        theta_tod[1:end-1],
        psi_tod,
    )
    hwp_cp = ConvolutionParams(
        nside_output=4,
        lstop=8,
        lmax=8,
        HWP=true,
        HWP_rpm=88.0,
    )
    @test_throws ArgumentError make_ring_angles(
        hwp_cp,
        ring_geometry(hwp_cp, ring),
        phi_tod,
        theta_tod,
        psi_tod,
    )
end


"""A scan that wanders over several rings, with a few samples per hit pixel and
a deliberately shuffled time order so the buffer has real work to do."""
function _multi_ring_tod(cp, rings, positions_per_ring, hits)
    phi = Float64[]
    theta = Float64[]
    psi = Float64[]
    for hit in 1:hits, ring in rings
        geometry = ring_geometry(cp, ring)
        for position in positions_per_ring
            position <= geometry.pixel_count || continue
            pixel = geometry.first_pixel + position - 1
            theta_pixel, phi_pixel = Scanvolver.pix2angRing(cp.resol, pixel)
            push!(theta, theta_pixel + 2.0e-4 * sin(0.7 * hit + position))
            push!(phi, phi_pixel + 2.0e-4 * cos(0.9 * hit - position))
            push!(psi, mod(0.23 + 1.3 * hit + 0.5 * position, 2pi))
        end
    end
    return phi, theta, psi
end

"""The samples of one pixel as an order-independent sorted list. Angles stores
the residual rotation rather than the raw pointing, so compare (alpha, beta,
gamma, psi), which is a bijective function of it."""
function _angle_samples(angle)
    return sort([
        (angle.α[i], angle.β[i], angle.γ[i], angle.ψ[i])
        for i in 1:angle.num
    ])
end

@testset "Ring scan buffer" begin
    cp = ConvolutionParams(nside_output=8, lstop=8, lmax=8, bnmax=2, κ=2)
    rings = 4:9
    phi, theta, psi = _multi_ring_tod(cp, rings, 1:3:20, 5)
    sample_count = length(phi)
    @test sample_count > 100

    if CUDA.functional()
        buffer = build_ring_scan_buffer(cp, phi, theta, psi)
        @test buffer isa RingScanBuffer
        @test buffer.nside == cp.nside_output
        @test buffer.first_ring == 1
        @test buffer.last_ring == ring_count(cp)
        @test buffer.processed_samples == sample_count
        @test buffer.stored_samples == sample_count
        @test length(buffer) == sample_count
        @test ring_scan_buffer_bytes(buffer) >= 3 * 8 * sample_count
        @test occursin("RingScanBuffer(nside=8", sprint(show, buffer))

        # Every sample must sit in the bucket of the pixel it points at, and
        # the buckets must be contiguous and cover the stored samples.
        @test buffer.pixel_offsets[1] == 1
        @test buffer.pixel_offsets[end] == sample_count + 1
        @test issorted(buffer.pixel_offsets)
        for bucket in 1:(length(buffer.pixel_offsets) - 1)
            pixel = buffer.first_pixel + bucket - 1
            for sample in
                buffer.pixel_offsets[bucket]:(buffer.pixel_offsets[bucket + 1] - 1)
                @test Scanvolver.ang2pixRing(
                    cp.resol,
                    buffer.theta[sample],
                    buffer.phi[sample],
                ) == pixel
            end
        end

        # The buffer must reproduce the one-pass result exactly.
        for ring in rings
            geometry = ring_geometry(cp, ring)
            direct_active, direct_angles = make_ring_angles(
                cp,
                geometry,
                phi,
                theta,
                psi,
            )
            buffered_active, buffered_angles = make_ring_angles(
                cp,
                geometry,
                buffer,
            )
            try
                @test Array(buffered_active.positions) ==
                    Array(direct_active.positions)
                @test buffered_active.ring_pixel_count ==
                    direct_active.ring_pixel_count
                @test buffered_active.complete == direct_active.complete
                @test length(buffered_angles) == length(direct_angles)
                for index in eachindex(direct_angles)
                    @test buffered_angles[index].num ==
                        direct_angles[index].num
                    @test buffered_angles[index].θ_pix ==
                        direct_angles[index].θ_pix
                    @test buffered_angles[index].φ_pix ==
                        direct_angles[index].φ_pix
                    # Order within a pixel is not preserved, the samples are.
                    @test _angle_samples(buffered_angles[index]) ==
                        _angle_samples(direct_angles[index])
                end
                @test ring_hit_count(buffer, ring) ==
                    sum(angle.num for angle in direct_angles)
            finally
                CUDA.unsafe_free!(direct_active.positions)
                CUDA.unsafe_free!(buffered_active.positions)
            end
        end

        # Rings the scan never visits.
        for ring in (1, ring_count(cp))
            active, angles = make_ring_angles(
                cp,
                ring_geometry(cp, ring),
                buffer,
            )
            @test active === nothing
            @test isempty(angles)
            @test ring_hit_count(buffer, ring) == 0
        end

        # Chunked ingestion, a capacity hint, and one chunk must agree.
        chunks = [
            (
                phi[start:min(start + 39, sample_count)],
                theta[start:min(start + 39, sample_count)],
                psi[start:min(start + 39, sample_count)],
            )
            for start in 1:40:sample_count
        ]
        progress = NamedTuple[]
        chunked = build_ring_scan_buffer(
            cp,
            chunks;
            capacity_hint=8,
            progress_callback=record -> push!(progress, record),
        )
        @test length(progress) == length(chunks)
        @test progress[end].processed_samples == sample_count
        @test progress[end].stored_samples == sample_count
        @test chunked.stored_samples == buffer.stored_samples
        @test chunked.pixel_offsets == buffer.pixel_offsets
        for ring in rings
            geometry = ring_geometry(cp, ring)
            a1, angles1 = make_ring_angles(cp, geometry, buffer)
            a2, angles2 = make_ring_angles(cp, geometry, chunked)
            try
                @test Array(a1.positions) == Array(a2.positions)
                for index in eachindex(angles1)
                    @test _angle_samples(angles1[index]) ==
                        _angle_samples(angles2[index])
                end
            finally
                CUDA.unsafe_free!(a1.positions)
                CUDA.unsafe_free!(a2.positions)
            end
        end

        # A narrow ring range keeps only its own samples but counts them all.
        narrow = build_ring_scan_buffer(cp, phi, theta, psi; rings=6:7)
        @test narrow.processed_samples == sample_count
        @test narrow.stored_samples < sample_count
        @test narrow.stored_samples ==
            ring_hit_count(buffer, 6) + ring_hit_count(buffer, 7)
        @test narrow.first_ring == 6
        @test narrow.last_ring == 7
        @test ring_hit_count(narrow, 6) == ring_hit_count(buffer, 6)
        @test_throws ArgumentError ring_hit_count(narrow, 5)
        @test_throws ArgumentError make_ring_angles(
            cp,
            ring_geometry(cp, 5),
            narrow,
        )
        # A narrow range must not pay for the whole map.
        @test length(narrow.pixel_offsets) <
            length(buffer.pixel_offsets)

        @test_throws DimensionMismatch build_ring_scan_buffer(
            cp,
            phi,
            theta[1:end - 1],
            psi,
        )
        @test_throws ArgumentError build_ring_scan_buffer(
            cp,
            phi,
            theta,
            psi;
            rings=5:4,
        )
        other = ConvolutionParams(nside_output=4, lstop=8, lmax=8)
        @test_throws ArgumentError make_ring_angles(
            other,
            ring_geometry(other, 5),
            buffer,
        )
        hwp = ConvolutionParams(
            nside_output=8,
            lstop=8,
            lmax=8,
            HWP=true,
            HWP_rpm=88.0,
        )
        @test_throws ArgumentError build_ring_scan_buffer(
            hwp,
            phi,
            theta,
            psi,
        )
    end
end


@testset "Active ring pixel gather" begin
    ring_pixel_count = 6
    sky_count = 2
    term_count = 9

    if CUDA.functional()
        @test_throws ArgumentError ActiveRingPixels(Int[], ring_pixel_count)
        @test_throws ArgumentError ActiveRingPixels([1, 2], 0)
        @test_throws ArgumentError ActiveRingPixels([0, 2], ring_pixel_count)
        @test_throws ArgumentError ActiveRingPixels(
            [2, ring_pixel_count + 1],
            ring_pixel_count,
        )
        @test_throws ArgumentError ActiveRingPixels([3, 3], ring_pixel_count)
        @test_throws ArgumentError ActiveRingPixels([4, 2], ring_pixel_count)
        @test_throws ArgumentError ActiveRingPixels(
            collect(1:(ring_pixel_count + 1)),
            ring_pixel_count,
        )

        partial = ActiveRingPixels([2, 4, 5], ring_pixel_count)
        complete = ActiveRingPixels(
            collect(1:ring_pixel_count),
            ring_pixel_count,
        )
        try
            @test length(partial) == 3
            @test partial.complete == false
            @test partial.ring_pixel_count == ring_pixel_count
            @test Array(partial.positions) == Int32[2, 4, 5]
            @test eltype(partial.positions) === Int32
            @test length(complete) == ring_pixel_count
            @test complete.complete == true
            @test occursin("3/6", sprint(show, partial))
            @test occursin("complete", sprint(show, complete))

            for precision in (Float32, Float64)
                full_host = Complex{precision}[
                    complex(
                        precision(sky + 10 * term + 100 * pixel),
                        precision(-sky - 3 * term + 7 * pixel),
                    )
                    for sky in 1:sky_count,
                        term in 1:term_count,
                        pixel in 1:ring_pixel_count
                ]
                full_device = nothing
                gathered = nothing
                aliased = nothing
                narrow = nothing
                narrow_gathered = nothing
                mismatched = nothing
                try
                    full_device = CuArray(full_host)

                    gathered = select_active_sky_ring_gpu(
                        full_device,
                        partial,
                    )
                    @test gathered !== full_device
                    @test size(gathered) == (sky_count, term_count, 3)
                    @test eltype(gathered) === Complex{precision}
                    @test Array(gathered) ==
                        full_host[:, :, [2, 4, 5]]

                    # A complete ring is returned unchanged: no kernel, no
                    # allocation, and the result aliases the input.
                    aliased = select_active_sky_ring_gpu(
                        full_device,
                        complete,
                    )
                    @test aliased === full_device

                    # The same ActiveRingPixels serves an l tile of a
                    # different width: only the pixel axis is touched.
                    narrow_host = full_host[:, 1:4, :]
                    narrow = CuArray(narrow_host)
                    narrow_gathered = select_active_sky_ring_gpu(
                        narrow,
                        partial,
                    )
                    @test size(narrow_gathered) == (sky_count, 4, 3)
                    @test Array(narrow_gathered) ==
                        narrow_host[:, :, [2, 4, 5]]

                    mismatched = CUDA.zeros(
                        Complex{precision},
                        sky_count,
                        term_count,
                        ring_pixel_count + 1,
                    )
                    @test_throws DimensionMismatch select_active_sky_ring_gpu(
                        mismatched,
                        partial,
                    )

                    # Releasing an aliased result must not free the ring tile.
                    free_active_sky_ring!(aliased, full_device)
                    aliased = nothing
                    @test Array(full_device) == full_host

                    free_active_sky_ring!(gathered, full_device)
                    gathered = nothing
                    @test Array(full_device) == full_host
                finally
                    narrow_gathered === nothing ||
                        CUDA.unsafe_free!(narrow_gathered)
                    narrow === nothing || CUDA.unsafe_free!(narrow)
                    mismatched === nothing ||
                        CUDA.unsafe_free!(mismatched)
                    gathered === nothing ||
                        CUDA.unsafe_free!(gathered)
                    full_device === nothing ||
                        CUDA.unsafe_free!(full_device)
                end
            end
        finally
            CUDA.unsafe_free!(partial.positions)
            CUDA.unsafe_free!(complete.positions)
        end
    else
        @test_throws ArgumentError ActiveRingPixels([1, 2], 4)
    end
end


@testset "Convolution tile GEMM" begin
    sky_count = 3
    term_count = 7
    pixel_count = 4
    beam_count = 2

    if CUDA.functional()
        for precision in (Float32, Float64)
            sky_host, effective_host =
                _synthetic_convolution_tile_inputs(
                    precision,
                    sky_count,
                    term_count,
                    pixel_count,
                    beam_count,
                    0.0,
                )
            initial_host = Complex{precision}[
                complex(
                    precision(0.5 * sky - 0.25 * beam),
                    precision(0.125 * pixel + 0.75 * scan_weight),
                )
                for sky in 1:sky_count,
                    beam in 1:beam_count,
                    pixel in 1:pixel_count,
                    scan_weight in 1:2
            ]

            sky_device = nothing
            effective_device = nothing
            accumulated = nothing
            second_sky = nothing
            second_effective = nothing
            bad_term = nothing
            bad_pixel = nothing
            bad_scan_weight = nothing
            bad_accumulated = nothing
            empty_sky = nothing
            empty_effective = nothing
            empty_accumulated = nothing
            mixed_accumulated = nothing
            try
                sky_device = CuArray(sky_host)
                effective_device = CuArray(effective_host)
                accumulated = CuArray(initial_host)

                returned = CUDA.@sync accumulate_convolution_tile_gemm_gpu!(
                    accumulated,
                    sky_device,
                    effective_device,
                )
                @test returned === accumulated
                @test size(accumulated) == (
                    sky_count,
                    beam_count,
                    pixel_count,
                    2,
                )

                expected = _convolution_tile_cpu_reference(
                    initial_host,
                    sky_host,
                    effective_host,
                )
                tolerance = precision == Float32 ? 1e-5 : 1e-13
                result = Array(accumulated)
                @test result ≈ expected rtol=tolerance atol=tolerance

                # The non-zero initial buffer must have been added to, not
                # overwritten: beta=1.
                @test !(result ≈ _convolution_tile_cpu_reference(
                    zero(initial_host),
                    sky_host,
                    effective_host,
                ))

                # A wrong conjugation anywhere must be detectable, so the
                # inputs and the result have substantial imaginary parts.
                @test maximum(abs, imag.(result)) > 0.5
                @test !(result ≈ _convolution_tile_cpu_reference(
                    initial_host,
                    conj.(sky_host),
                    effective_host,
                ))
                @test !(result ≈ _convolution_tile_cpu_reference(
                    initial_host,
                    sky_host,
                    conj.(effective_host),
                ))
                @test !(result ≈ conj.(_convolution_tile_cpu_reference(
                    initial_host,
                    sky_host,
                    effective_host,
                )))

                # A second l tile accumulates into the same buffer.
                second_sky_host, second_effective_host =
                    _synthetic_convolution_tile_inputs(
                        precision,
                        sky_count,
                        term_count,
                        pixel_count,
                        beam_count,
                        1.7,
                    )
                second_sky = CuArray(second_sky_host)
                second_effective = CuArray(second_effective_host)
                CUDA.@sync accumulate_convolution_tile_gemm_gpu!(
                    accumulated,
                    second_sky,
                    second_effective,
                )
                two_tile_expected = _convolution_tile_cpu_reference(
                    expected,
                    second_sky_host,
                    second_effective_host,
                )
                @test Array(accumulated) ≈
                    two_tile_expected rtol=tolerance atol=tolerance

                bad_term = CUDA.zeros(
                    Complex{precision},
                    term_count + 1,
                    beam_count,
                    pixel_count,
                    2,
                )
                @test_throws DimensionMismatch accumulate_convolution_tile_gemm_gpu!(
                    accumulated,
                    sky_device,
                    bad_term,
                )
                bad_pixel = CUDA.zeros(
                    Complex{precision},
                    term_count,
                    beam_count,
                    pixel_count + 1,
                    2,
                )
                @test_throws DimensionMismatch accumulate_convolution_tile_gemm_gpu!(
                    accumulated,
                    sky_device,
                    bad_pixel,
                )
                bad_scan_weight = CUDA.zeros(
                    Complex{precision},
                    term_count,
                    beam_count,
                    pixel_count,
                    3,
                )
                @test_throws DimensionMismatch accumulate_convolution_tile_gemm_gpu!(
                    accumulated,
                    sky_device,
                    bad_scan_weight,
                )
                bad_accumulated = CUDA.zeros(
                    Complex{precision},
                    sky_count,
                    beam_count + 1,
                    pixel_count,
                    2,
                )
                @test_throws DimensionMismatch accumulate_convolution_tile_gemm_gpu!(
                    bad_accumulated,
                    sky_device,
                    effective_device,
                )

                empty_sky = CUDA.zeros(
                    Complex{precision},
                    sky_count,
                    term_count,
                    0,
                )
                empty_effective = CUDA.zeros(
                    Complex{precision},
                    term_count,
                    beam_count,
                    0,
                    2,
                )
                empty_accumulated = CUDA.zeros(
                    Complex{precision},
                    sky_count,
                    beam_count,
                    0,
                    2,
                )
                @test_throws ArgumentError accumulate_convolution_tile_gemm_gpu!(
                    empty_accumulated,
                    empty_sky,
                    empty_effective,
                )

                # Mixed precision is rejected by dispatch.
                other = precision == Float32 ? Float64 : Float32
                mixed_accumulated = CUDA.zeros(
                    Complex{other},
                    sky_count,
                    beam_count,
                    pixel_count,
                    2,
                )
                @test_throws MethodError accumulate_convolution_tile_gemm_gpu!(
                    mixed_accumulated,
                    sky_device,
                    effective_device,
                )
            finally
                for array in (
                    sky_device,
                    effective_device,
                    accumulated,
                    second_sky,
                    second_effective,
                    bad_term,
                    bad_pixel,
                    bad_scan_weight,
                    bad_accumulated,
                    empty_sky,
                    empty_effective,
                    empty_accumulated,
                    mixed_accumulated,
                )
                    array === nothing || CUDA.unsafe_free!(array)
                end
            end
        end
    end
end

@testset "Convolution tile integration" begin
    cp = ConvolutionParams(
        nside_output=2,
        lstart=0,
        lstop=5,
        lmax=5,
        bnmax=2,
        κ=2,
        numofsky=2,
        numofbeam=2,
    )
    pixel_count = 3
    angles = [
        _synthetic_scan_moment_angles(
            cp;
            pixel=pixel,
            hit_count=8 + pixel,
        )
        for pixel in 1:pixel_count
    ]
    beta_max = 1.05 * maximum(
        maximum(angle.β)
        for angle in angles
    )
    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    raw_alms = [
        ComplexF64[
            complex(
                sin(0.2 * spin + 0.3 * coefficient + sky),
                cos(0.4 * spin - 0.1 * coefficient + 0.5 * sky),
            )
            for spin in 1:3, coefficient in 1:coefficient_count
        ]
        for sky in 1:cp.numofsky
    ]
    raw_blms = [
        ComplexF64[
            complex(
                sin(0.3 * spin + 0.2 * coefficient + beam),
                cos(0.4 * spin - 0.1 * coefficient + 0.5 * beam),
            )
            for spin in 1:3, coefficient in 1:coefficient_count
        ]
        for beam in 1:cp.numofbeam
    ]
    phi0 = 0.17
    phi_pixels = [
        phi0 + 2pi * (pixel_index - 1) / pixel_count
        for pixel_index in 1:pixel_count
    ]
    theta_ring = 0.73

    if CUDA.functional()
        for precision in (Float32, Float64)
            plan = ScanMomentPlan(
                kappa=cp.κ,
                nmax=calc_nmax(cp, cp.lstop),
                beta_order=10,
                beta_max=beta_max,
                precision=precision,
            )
            table = build_wigner_chebyshev_table_cpu(cp, plan)
            device_table = upload_wigner_table_gpu(table)
            prepared_alm = prepare_alm(
                cp,
                raw_alms;
                precision=precision,
            )
            prepared_blm = prepare_blm(
                cp,
                raw_blms;
                precision=precision,
            )
            alm_gpu = upload_alm_gpu(prepared_alm)
            moments_device = CUDA.@sync scan_moments_gemm_gpu(
                plan,
                angles,
            )
            moments_host = cat(
                (scan_moments_cpu(plan, angle) for angle in angles)...;
                dims=5,
            )
            ringd = CUDA.@sync _ringd_gpu(
                cp,
                theta_ring;
                precision=precision,
            )

            accumulated = nothing
            sky_ring = nothing
            effective = nothing
            try
                accumulated = CUDA.zeros(
                    Complex{precision},
                    cp.numofsky,
                    cp.numofbeam,
                    pixel_count,
                    2,
                )
                expected = zeros(
                    Complex{precision},
                    cp.numofsky,
                    cp.numofbeam,
                    pixel_count,
                    2,
                )
                cpu_expected = copy(expected)

                # Two l tiles covering 1:5 accumulate into one buffer.
                for (l_first, l_last) in ((1, 3), (4, 5))
                    sky_ring = CUDA.@sync build_sky_ring_gpu(
                        cp,
                        ringd,
                        alm_gpu,
                        phi_pixels,
                        l_first,
                        l_last,
                    )
                    effective = CUDA.@sync _effective_beam_gpu(
                        device_table,
                        moments_device,
                        prepared_blm,
                        l_first,
                        l_last,
                    )
                    # The two producers must agree on the (spin, M, l) axis.
                    @test size(sky_ring, 2) == size(effective, 1)
                    @test size(sky_ring, 3) == size(effective, 3)

                    expected = _convolution_tile_cpu_reference(
                        expected,
                        Array(sky_ring),
                        Array(effective),
                    )
                    cpu_expected = _convolution_tile_cpu_reference(
                        cpu_expected,
                        _sky_ring_cpu_reference(
                            cp,
                            Array(ringd),
                            prepared_alm,
                            phi_pixels,
                            l_first,
                            l_last,
                        ),
                        _effective_beam_cpu_reference(
                            table,
                            moments_host,
                            prepared_blm,
                            l_first,
                            l_last,
                        ),
                    )

                    CUDA.@sync accumulate_convolution_tile_gemm_gpu!(
                        accumulated,
                        sky_ring,
                        effective,
                    )
                    CUDA.unsafe_free!(sky_ring)
                    sky_ring = nothing
                    CUDA.unsafe_free!(effective)
                    effective = nothing
                end

                result = Array(accumulated)
                @test size(result) == (
                    cp.numofsky,
                    cp.numofbeam,
                    pixel_count,
                    2,
                )
                @test all(isfinite, result)
                @test !any(iszero, result)

                # Measured max relative error against these oracles is
                # 1.7e-7 (Float32) and 7.3e-16 (Float64); the end-to-end
                # CPU path differs by 6.1e-7 and 1.5e-15.
                device_tolerance = precision == Float32 ? 1e-5 : 1e-13
                @test result ≈ expected rtol=device_tolerance atol=device_tolerance

                # Fully independent CPU path: CPU sky ring and CPU effective
                # beam contracted on the CPU.
                end_to_end_tolerance =
                    precision == Float32 ? 1e-5 : 1e-11
                @test result ≈ cpu_expected rtol=end_to_end_tolerance atol=end_to_end_tolerance

                # The spin weight (1,1/2,1/2) is applied exactly once, in
                # build_sky_ring_gpu. Reapplying it would change the result.
                @test !(result ≈ cpu_expected .* 0.5)
            finally
                sky_ring === nothing || CUDA.unsafe_free!(sky_ring)
                effective === nothing || CUDA.unsafe_free!(effective)
                accumulated === nothing ||
                    CUDA.unsafe_free!(accumulated)
                CUDA.unsafe_free!(ringd)
                CUDA.unsafe_free!(moments_device)
                CUDA.unsafe_free!(alm_gpu.values)
                CUDA.unsafe_free!(device_table.coefficients)
                CUDA.unsafe_free!(device_table.pair_lookup)
            end
        end
    end
end


@testset "Convolution partial ring integration" begin
    cp = ConvolutionParams(
        nside_output=2,
        lstart=0,
        lstop=5,
        lmax=5,
        bnmax=2,
        κ=2,
        numofsky=2,
        numofbeam=2,
    )
    ring_pixel_count = 5
    active_positions = [2, 4, 5]
    # Only the hit pixels get scan geometry, so the effective beam is built
    # for three pixels while the sky ring FFT must cover all five.
    angles = [
        _synthetic_scan_moment_angles(
            cp;
            pixel=pixel,
            hit_count=8 + pixel,
        )
        for pixel in active_positions
    ]
    beta_max = 1.05 * maximum(
        maximum(angle.β)
        for angle in angles
    )
    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    raw_alms = [
        ComplexF64[
            complex(
                sin(0.2 * spin + 0.3 * coefficient + sky),
                cos(0.4 * spin - 0.1 * coefficient + 0.5 * sky),
            )
            for spin in 1:3, coefficient in 1:coefficient_count
        ]
        for sky in 1:cp.numofsky
    ]
    raw_blms = [
        ComplexF64[
            complex(
                sin(0.3 * spin + 0.2 * coefficient + beam),
                cos(0.4 * spin - 0.1 * coefficient + 0.5 * beam),
            )
            for spin in 1:3, coefficient in 1:coefficient_count
        ]
        for beam in 1:cp.numofbeam
    ]
    phi_pixels = [
        0.17 + 2pi * (pixel_index - 1) / ring_pixel_count
        for pixel_index in 1:ring_pixel_count
    ]
    theta_ring = 0.73

    if CUDA.functional()
        for precision in (Float32, Float64)
            plan = ScanMomentPlan(
                kappa=cp.κ,
                nmax=calc_nmax(cp, cp.lstop),
                beta_order=10,
                beta_max=beta_max,
                precision=precision,
            )
            table = build_wigner_chebyshev_table_cpu(cp, plan)
            device_table = upload_wigner_table_gpu(table)
            prepared_alm = prepare_alm(
                cp,
                raw_alms;
                precision=precision,
            )
            prepared_blm = prepare_blm(
                cp,
                raw_blms;
                precision=precision,
            )
            alm_gpu = upload_alm_gpu(prepared_alm)
            moments_device = CUDA.@sync scan_moments_gemm_gpu(
                plan,
                angles,
            )
            moments_host = cat(
                (scan_moments_cpu(plan, angle) for angle in angles)...;
                dims=5,
            )
            ringd = CUDA.@sync _ringd_gpu(
                cp,
                theta_ring;
                precision=precision,
            )
            active = ActiveRingPixels(
                active_positions,
                ring_pixel_count,
            )

            accumulated = nothing
            sky_ring = nothing
            active_sky_ring = nothing
            effective = nothing
            try
                accumulated = CUDA.zeros(
                    Complex{precision},
                    cp.numofsky,
                    cp.numofbeam,
                    length(active_positions),
                    2,
                )
                expected = zeros(
                    Complex{precision},
                    cp.numofsky,
                    cp.numofbeam,
                    length(active_positions),
                    2,
                )

                # One ActiveRingPixels drives both l tiles.
                for (l_first, l_last) in ((1, 3), (4, 5))
                    sky_ring = CUDA.@sync build_sky_ring_gpu(
                        cp,
                        ringd,
                        alm_gpu,
                        phi_pixels,
                        l_first,
                        l_last,
                    )
                    @test size(sky_ring, 3) == ring_pixel_count

                    active_sky_ring = select_active_sky_ring_gpu(
                        sky_ring,
                        active,
                    )
                    @test size(active_sky_ring, 3) ==
                        length(active_positions)

                    effective = CUDA.@sync _effective_beam_gpu(
                        device_table,
                        moments_device,
                        prepared_blm,
                        l_first,
                        l_last,
                    )
                    @test size(effective, 3) == length(active_positions)
                    @test size(active_sky_ring, 2) == size(effective, 1)

                    CUDA.@sync accumulate_convolution_tile_gemm_gpu!(
                        accumulated,
                        active_sky_ring,
                        effective,
                    )

                    # Independent CPU path: full-ring reference restricted to
                    # the hit positions.
                    expected = _convolution_tile_cpu_reference(
                        expected,
                        _sky_ring_cpu_reference(
                            cp,
                            Array(ringd),
                            prepared_alm,
                            phi_pixels,
                            l_first,
                            l_last,
                        )[:, :, active_positions],
                        _effective_beam_cpu_reference(
                            table,
                            moments_host,
                            prepared_blm,
                            l_first,
                            l_last,
                        ),
                    )

                    free_active_sky_ring!(active_sky_ring, sky_ring)
                    active_sky_ring = nothing
                    CUDA.unsafe_free!(sky_ring)
                    sky_ring = nothing
                    CUDA.unsafe_free!(effective)
                    effective = nothing
                end

                result = Array(accumulated)
                @test all(isfinite, result)
                @test !any(iszero, result)
                tolerance = precision == Float32 ? 1e-5 : 1e-11
                @test result ≈ expected rtol=tolerance atol=tolerance

                # Gathering the wrong pixels must be detectable: the result
                # differs from the same contraction on a shifted selection.
                shifted = _convolution_tile_cpu_reference(
                    zero(expected),
                    _sky_ring_cpu_reference(
                        cp,
                        Array(ringd),
                        prepared_alm,
                        phi_pixels,
                        1,
                        5,
                    )[:, :, [1, 3, 4]],
                    _effective_beam_cpu_reference(
                        table,
                        moments_host,
                        prepared_blm,
                        1,
                        5,
                    ),
                )
                @test !(result ≈ shifted)
            finally
                active_sky_ring === nothing || free_active_sky_ring!(
                    active_sky_ring,
                    sky_ring === nothing ? active_sky_ring : sky_ring,
                )
                sky_ring === nothing || CUDA.unsafe_free!(sky_ring)
                effective === nothing || CUDA.unsafe_free!(effective)
                accumulated === nothing ||
                    CUDA.unsafe_free!(accumulated)
                CUDA.unsafe_free!(active.positions)
                CUDA.unsafe_free!(ringd)
                CUDA.unsafe_free!(moments_device)
                CUDA.unsafe_free!(alm_gpu.values)
                CUDA.unsafe_free!(device_table.coefficients)
                CUDA.unsafe_free!(device_table.pair_lookup)
            end
        end
    end
end


function _ring_scan_tod(cp, geometry, positions, hits_per_pixel)
    phi = Float64[]
    theta = Float64[]
    psi = Float64[]
    for position in positions, hit in 1:hits_per_pixel
        pixel = geometry.first_pixel + position - 1
        theta_pixel, phi_pixel = Scanvolver.pix2angRing(cp.resol, pixel)
        push!(theta, theta_pixel + 2.0e-4 * sin(0.7 * hit + position))
        push!(phi, phi_pixel + 2.0e-4 * cos(0.9 * hit - position))
        push!(psi, mod(0.23 + 1.3 * hit + 0.5 * position, 2pi))
    end
    return phi, theta, psi
end

@testset "Ring memory model" begin
    cp = ConvolutionParams(
        nside_output=8,
        lstart=0,
        lstop=10,
        lmax=10,
        bnmax=2,
        κ=3,
        numofsky=2,
        numofbeam=3,
    )
    full_range = cp.lstop - cp.lstart + 1

    estimate = ring_memory_bytes(cp, 32, 16, 4)
    @test estimate.persistent ==
        estimate.ringd_plan + estimate.ringd + estimate.accumulated
    @test estimate.total == estimate.persistent + estimate.tile_peak
    @test estimate.tile_peak >= estimate.effective_beam

    # Everything that scales with the tile is linear in it, which is what
    # the pipelines' tile sizing relies on.
    unit = ring_memory_bytes(cp, 32, 16, 1)
    for tile in (2, 3, 5, 8)
        scaled = ring_memory_bytes(cp, 32, 16, tile)
        @test scaled.tile_peak == tile * unit.tile_peak
        @test scaled.effective_kernel == tile * unit.effective_kernel
        @test scaled.sky_ring == tile * unit.sky_ring
        @test scaled.persistent == unit.persistent
    end

    # A complete ring needs no gathered copy.
    @test ring_memory_bytes(cp, 32, 32, 4; complete_ring=true).active_sky_ring == 0
    @test ring_memory_bytes(cp, 32, 32, 4).active_sky_ring > 0
    @test ring_memory_bytes(cp, 32, 16, 4; precision=Float32).ringd ==
        ring_memory_bytes(cp, 32, 16, 4; precision=Float64).ringd ÷ 2

    @test_throws ArgumentError ring_memory_bytes(cp, 32, 16, 0)
    @test_throws ArgumentError ring_memory_bytes(cp, 32, 33, 4)
    @test_throws ArgumentError ring_memory_bytes(cp, 0, 1, 4)
    @test_throws ArgumentError ring_memory_bytes(cp, 32, 16, 4; precision=Int)

    # The Int32 ceiling. At a production shape it binds well below the full
    # range; the width it reports must be the largest that still indexes.
    big = ConvolutionParams(
        nside_output=512,
        lstart=0,
        lstop=1200,
        lmax=1200,
        bnmax=120,
        κ=10,
        numofsky=5,
        numofbeam=5,
    )
    ceiling = max_indexable_l_tile(big, 2048)
    @test 1 <= ceiling < 1201
    M_count = 2 * calc_Mmax(big, big.lstop) + 1
    delta_count = 2 * big.κ + 1
    per_tile = maximum((
        2 * delta_count * 2048 * M_count,
        3 * big.numofbeam * delta_count * M_count,
        2 * 3 * M_count * big.numofbeam * 2048,
        3 * big.numofbeam * 2048 * M_count,
    ))
    @test per_tile * ceiling <= typemax(Int32)
    @test per_tile * (ceiling + 1) > typemax(Int32)
    @test max_indexable_l_tile(cp, 16) == full_range
    @test_throws ArgumentError max_indexable_l_tile(cp, 0)

    if CUDA.functional()
        # The estimate must be the actual allocation, byte for byte.
        ring = 7
        geometry = ring_geometry(cp, ring)
        positions = 1:2:geometry.pixel_count
        phi, theta, psi = _ring_scan_tod(cp, geometry, positions, 6)
        active, angles = make_ring_angles(cp, geometry, phi, theta, psi)
        beta_max = 1.05 * maximum(
            maximum(angle.β)
            for angle in angles
        )
        plan = ScanMomentPlan(
            kappa=cp.κ,
            nmax=calc_nmax(cp, cp.lstop),
            beta_order=8,
            beta_max=beta_max,
            precision=Float64,
        )
        device_table = upload_wigner_table_gpu(
            build_wigner_chebyshev_table_cpu(cp, plan),
        )
        coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
        raw_alms = [
            ComplexF64[
                complex(sin(0.2 * s + 0.3 * c + k), cos(0.4 * s - 0.1 * c))
                for s in 1:3, c in 1:coefficient_count
            ]
            for k in 1:cp.numofsky
        ]
        raw_blms = [
            ComplexF64[
                complex(sin(0.3 * s + 0.2 * c + b), cos(0.4 * s - 0.1 * c))
                for s in 1:3, c in 1:coefficient_count
            ]
            for b in 1:cp.numofbeam
        ]
        alm_gpu = upload_alm_gpu(prepare_alm(cp, raw_alms; precision=Float64))
        prepared_blm = prepare_blm(cp, raw_blms; precision=Float64)
        moments = scan_moments_gpu(plan, angles)
        tile = 4
        measured = ring_memory_bytes(
            cp,
            geometry.pixel_count,
            length(active),
            tile;
            precision=Float64,
            complete_ring=active.complete,
        )
        ringd = nothing
        accumulated = nothing
        kernel = nothing
        effective = nothing
        sky_ring = nothing
        active_sky_ring = nothing
        try
            ringd = _ringd_gpu(
                cp,
                geometry.theta;
                precision=Float64,
            )
            accumulated = CUDA.zeros(
                ComplexF64,
                cp.numofsky,
                cp.numofbeam,
                length(active),
                2,
            )
            kernel = build_effective_kernel_gpu(
                device_table,
                moments,
                1,
                tile,
            )
            effective = _effective_beam_gpu(
                device_table,
                moments,
                prepared_blm,
                1,
                tile,
            )
            sky_ring = build_sky_ring_gpu(
                cp,
                ringd,
                alm_gpu,
                geometry.phi_pixels,
                1,
                tile,
            )
            active_sky_ring = select_active_sky_ring_gpu(sky_ring, active)

            @test Int(measured.ringd) == sizeof(ringd)
            @test Int(measured.accumulated) == sizeof(accumulated)
            @test Int(measured.effective_kernel) == sizeof(kernel)
            @test Int(measured.effective_beam) == sizeof(effective)
            @test Int(measured.sky_ring) == sizeof(sky_ring)
            @test Int(measured.active_sky_ring) ==
                (active_sky_ring === sky_ring ? 0 : sizeof(active_sky_ring))
        finally
            active_sky_ring === nothing || sky_ring === nothing ||
                free_active_sky_ring!(active_sky_ring, sky_ring)
            sky_ring === nothing || CUDA.unsafe_free!(sky_ring)
            effective === nothing || CUDA.unsafe_free!(effective)
            kernel === nothing || CUDA.unsafe_free!(kernel)
            accumulated === nothing || CUDA.unsafe_free!(accumulated)
            ringd === nothing || CUDA.unsafe_free!(ringd)
            CUDA.unsafe_free!(moments)
            CUDA.unsafe_free!(alm_gpu.values)
            CUDA.unsafe_free!(device_table.coefficients)
            CUDA.unsafe_free!(device_table.pair_lookup)
            CUDA.unsafe_free!(active.positions)
        end
    end
end


@testset "RingD context reuse" begin
    cp = ConvolutionParams(
        nside_output=8,
        lstart=0,
        lstop=8,
        lmax=8,
        bnmax=2,
        κ=2,
        numofsky=1,
        numofbeam=1,
    )

    if CUDA.functional()
        for precision in (Float32, Float64)
            context = ringd_context(cp; precision=precision)
            try
                @test context isa RingDContext{precision}
                @test context.lstart == cp.lstart
                @test context.lstop == cp.lstop
                @test context.Mmax == calc_Mmax(cp, cp.lstop)
                @test eltype(context) === Complex{precision}
                @test size(context.ringd) == (
                    2 * cp.lstop + 1,
                    2 * calc_Mmax(cp, cp.lstop) + 1,
                    cp.lstop - cp.lstart + 1,
                )
                @test occursin("RingDContext{$precision}", sprint(show, context))

                # Refilling must give exactly what a fresh context gives, and a
                # second ring must leave nothing of the first behind.
                for ring in (3, 9, 15)
                    theta = ring_geometry(cp, ring).theta
                    reused = ringd_blocks_recursion_gpu!(context, cp, theta)
                    @test reused === context.ringd
                    fresh = _ringd_gpu(
                        cp,
                        theta;
                        precision=precision,
                    )
                    try
                        @test Array(reused) == Array(fresh)
                    finally
                        CUDA.unsafe_free!(fresh)
                    end
                end
            finally
                free_ringd_context!(context)
            end
        end

        wrong = ConvolutionParams(
            nside_output=8,
            lstart=0,
            lstop=6,
            lmax=8,
            bnmax=2,
            κ=2,
        )
        context = ringd_context(cp)
        try
            @test_throws ArgumentError ringd_blocks_recursion_gpu!(
                context,
                wrong,
                0.5,
            )
        finally
            free_ringd_context!(context)
        end
    end
end


@testset "Ring convolution" begin
    cp = ConvolutionParams(
        nside_output=4,
        lstart=0,
        lstop=5,
        lmax=5,
        bnmax=2,
        κ=2,
        numofsky=2,
        numofbeam=2,
    )
    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    raw_alms = [
        ComplexF64[
            complex(
                sin(0.2 * spin + 0.3 * coefficient + sky),
                cos(0.4 * spin - 0.1 * coefficient + 0.5 * sky),
            )
            for spin in 1:3, coefficient in 1:coefficient_count
        ]
        for sky in 1:cp.numofsky
    ]
    raw_blms = [
        ComplexF64[
            complex(
                sin(0.3 * spin + 0.2 * coefficient + beam),
                cos(0.4 * spin - 0.1 * coefficient + 0.5 * beam),
            )
            for spin in 1:3, coefficient in 1:coefficient_count
        ]
        for beam in 1:cp.numofbeam
    ]

    if CUDA.functional()
        # Ring 5 has 16 pixels and no FFT aliasing at lstop=5; ring 1 has 4
        # pixels, so |m| up to 5 aliases into 4 bins and every pixel is hit.
        for (ring, hit_positions) in (
            (5, [2, 5, 9, 14]),
            (1, [1, 2, 3, 4]),
        )
            geometry = ring_geometry(cp, ring)
            phi_tod, theta_tod, psi_tod = _ring_scan_tod(
                cp,
                geometry,
                hit_positions,
                6,
            )

            for precision in (Float32, Float64)
                active, angles = make_ring_angles(
                    cp,
                    geometry,
                    phi_tod,
                    theta_tod,
                    psi_tod,
                )
                beta_max = 1.05 * maximum(
                    maximum(angle.β)
                    for angle in angles
                )
                plan = ScanMomentPlan(
                    kappa=cp.κ,
                    nmax=calc_nmax(cp, cp.lstop),
                    beta_order=10,
                    beta_max=beta_max,
                    precision=precision,
                )
                table = build_wigner_chebyshev_table_cpu(cp, plan)
                device_table = upload_wigner_table_gpu(table)
                prepared_alm = prepare_alm(
                    cp,
                    raw_alms;
                    precision=precision,
                )
                prepared_blm = prepare_blm(
                    cp,
                    raw_blms;
                    precision=precision,
                )
                alm_gpu = upload_alm_gpu(prepared_alm)
                ringd = _ringd_gpu(cp, geometry.theta; precision=precision)
                ringd_host = Array(ringd)
                CUDA.unsafe_free!(ringd)

                results = RingConvolution[]
                try
                    @test active.complete == (ring == 1)
                    @test length(angles) == length(hit_positions)

                    expected = _ring_convolution_cpu_reference(
                        cp,
                        geometry,
                        hit_positions,
                        angles,
                        table,
                        prepared_alm,
                        prepared_blm,
                        ringd_host,
                        plan,
                        cp.lstop - cp.lstart + 1,
                    )
                    tolerance = precision == Float32 ? 1e-4 : 1e-11

                    # The same ring must come out identically however the l
                    # range is split into tiles.
                    for l_tile in (6, 3, 2, 1)
                        result = _convolve_ring_stages(
                            cp, geometry, active, angles, alm_gpu, prepared_blm,
                            device_table, plan; l_tile=l_tile,
                        )
                        push!(results, result)
                        @test result isa RingConvolution{precision}
                        @test result.ring == ring
                        @test result.positions == hit_positions
                        @test result.pixels == [
                            geometry.first_pixel + position - 1
                            for position in hit_positions
                        ]
                        @test result.ring_pixel_count ==
                            geometry.pixel_count
                        @test length(result) == length(hit_positions)
                        @test size(result.accumulated) == (
                            cp.numofsky,
                            cp.numofbeam,
                            length(hit_positions),
                            2,
                        )
                        values = Array(result.accumulated)
                        @test all(isfinite, values)
                        @test !any(iszero, values)
                        @test values ≈
                            expected rtol=tolerance atol=tolerance
                    end
                    @test Array(results[1].accumulated) ≈
                        Array(results[end].accumulated) rtol=tolerance atol=tolerance
                    @test occursin(
                        "ring=$ring",
                        sprint(show, results[1]),
                    )

                    # Direct scan moments must give the same ring.
                    direct = _convolve_ring_stages(
                        cp, geometry, active, angles, alm_gpu, prepared_blm,
                        device_table, plan; l_tile=3, moment_method=:direct,
                    )
                    push!(results, direct)
                    @test Array(direct.accumulated) ≈
                        expected rtol=tolerance atol=tolerance
                finally
                    for result in results
                        CUDA.unsafe_free!(result.accumulated)
                    end
                    CUDA.unsafe_free!(active.positions)
                    CUDA.unsafe_free!(alm_gpu.values)
                    CUDA.unsafe_free!(device_table.coefficients)
                    CUDA.unsafe_free!(device_table.pair_lookup)
                end
            end
        end
    end
end


function _angles_with_psi(cp, pixel, psi)
    theta_pixel, phi_pixel = Scanvolver.pix2angRing(cp.resol, pixel)
    hit_count = length(psi)
    return Angles(
        cp;
        pix=pixel,
        φ=Float64[
            phi_pixel + 1.5e-4 * cos(0.7 * hit)
            for hit in 1:hit_count
        ],
        θ=Float64[
            theta_pixel + 1.5e-4 * sin(0.9 * hit)
            for hit in 1:hit_count
        ],
        ψ=Float64[Float64(value) for value in psi],
    )
end

# The detector model the binning matrix encodes.
_ideal_signal(I, Q, U, psi) = I + Q * cos(2 * psi) + U * sin(2 * psi)

function _synthetic_ring_convolution(
    ::Type{T},
    cp,
    geometry,
    positions,
    angles,
    stokes,
) where {T}
    active_count = length(positions)
    accumulated = zeros(
        Complex{T},
        cp.numofsky,
        cp.numofbeam,
        active_count,
        2,
    )
    for active_index in 1:active_count
        angle = angles[active_index]
        for beam_index in 1:cp.numofbeam,
            sky_index in 1:cp.numofsky
            I, Q, U = stokes[sky_index, beam_index, active_index]
            signal = [
                _ideal_signal(I, Q, U, psi)
                for psi in angle.ψ
            ]
            accumulated[sky_index, beam_index, active_index, 1] =
                sum(signal) / angle.num
            accumulated[sky_index, beam_index, active_index, 2] =
                sum(
                    signal[hit] * cis(2 * angle.ψ[hit])
                    for hit in 1:angle.num
                ) / angle.num
        end
    end
    return RingConvolution{T}(
        geometry.ring,
        geometry.first_pixel,
        geometry.pixel_count,
        positions,
        [geometry.first_pixel + position - 1 for position in positions],
        CuArray(accumulated),
    )
end



@testset "Ring Stokes solve" begin
    cp = ConvolutionParams(
        nside_output=4,
        lstop=5,
        lmax=5,
        numofsky=2,
        numofbeam=2,
    )

    if CUDA.functional()
        geometry = ring_geometry(cp, 5)
        positions = [2, 7, 12]
        # Well spread polarization angles keep the 3x3 system conditioned.
        psi_sets = [
            [0.0, 0.41, 0.83, 1.24, 1.66, 2.07, 2.49],
            [0.13, 0.62, 1.11, 1.60, 2.09, 2.58, 3.07],
            [0.27, 0.71, 1.15, 1.59, 2.03, 2.47, 2.91],
        ]
        angles = [
            _angles_with_psi(
                cp,
                geometry.first_pixel + position - 1,
                psi_sets[active_index],
            )
            for (active_index, position) in pairs(positions)
        ]

        # Distinct Q and U per sky/beam/pixel, including pure-Q and pure-U
        # pixels so a swapped or sign-flipped component cannot pass.
        stokes = Array{NTuple{3,Float64},3}(undef, 2, 2, 3)
        stokes[1, 1, 1] = (1.5, 0.3, -0.7)
        stokes[2, 1, 1] = (-0.4, 0.9, 0.2)
        stokes[1, 2, 1] = (2.0, -1.1, 0.55)
        stokes[2, 2, 1] = (0.25, 0.0, 0.8)
        stokes[1, 1, 2] = (0.7, 0.6, 0.0)
        stokes[2, 1, 2] = (0.0, -0.5, -0.25)
        stokes[1, 2, 2] = (-1.3, 0.15, 0.95)
        stokes[2, 2, 2] = (3.1, -0.75, -0.35)
        stokes[1, 1, 3] = (0.9, 0.45, 0.65)
        stokes[2, 1, 3] = (1.2, -0.2, 0.1)
        stokes[1, 2, 3] = (-0.6, 0.85, -0.9)
        stokes[2, 2, 3] = (0.05, 0.35, 0.4)

        for precision in (Float32, Float64)
            result = _synthetic_ring_convolution(
                precision,
                cp,
                geometry,
                positions,
                angles,
                stokes,
            )
            try
                solved = ring_stokes(result, angles; precision=precision)
                @test size(solved) == (2, 2, 3, length(positions))
                @test eltype(solved) === precision

                tolerance = precision == Float32 ? 2e-5 : 1e-12
                for active_index in eachindex(positions)
                    for beam_index in 1:2, sky_index in 1:2
                        I, Q, U = stokes[
                            sky_index,
                            beam_index,
                            active_index,
                        ]
                        @test solved[
                            sky_index, beam_index, 1, active_index,
                        ] ≈ I atol=tolerance rtol=tolerance
                        @test solved[
                            sky_index, beam_index, 2, active_index,
                        ] ≈ Q atol=tolerance rtol=tolerance
                        @test solved[
                            sky_index, beam_index, 3, active_index,
                        ] ≈ U atol=tolerance rtol=tolerance
                    end
                end
            finally
                CUDA.unsafe_free!(result.accumulated)
            end
        end

        # A pixel scanned at a single polarization angle cannot separate Q
        # from U: the system is rank deficient and the pixel is unseen.
        degenerate_angles = [
            _angles_with_psi(
                cp,
                geometry.first_pixel + positions[1] - 1,
                fill(0.37, 6),
            ),
            angles[2],
            angles[3],
        ]
        degenerate = _synthetic_ring_convolution(
            Float64,
            cp,
            geometry,
            positions,
            degenerate_angles,
            stokes,
        )
        try
            solved = ring_stokes(degenerate, degenerate_angles)
            @test all(isnan, solved[:, :, :, 1])
            @test all(isfinite, solved[:, :, :, 2])
            @test all(iszero, ring_stokes(
                degenerate,
                degenerate_angles;
                unseen=0.0,
            )[:, :, :, 1])

            # rcond=0 disables the rejection.
            permissive = ring_stokes(
                degenerate,
                degenerate_angles;
                rcond=0.0,
            )
            @test !all(isnan, permissive[:, :, :, 2])

            @test_throws DimensionMismatch ring_stokes(
                degenerate,
                degenerate_angles[1:2],
            )
            @test_throws ArgumentError ring_stokes(
                degenerate,
                degenerate_angles;
                rcond=-1.0,
            )
            @test_throws ArgumentError ring_stokes(
                degenerate,
                degenerate_angles;
                precision=Int,
            )
        finally
            CUDA.unsafe_free!(degenerate.accumulated)
        end

        # The production default rejects a finite but poorly conditioned
        # polarization solve. A deliberately permissive diagnostic setting
        # can still retain it.
        marginal_angles = copy(angles)
        marginal_angles[1] = _angles_with_psi(
            cp,
            geometry.first_pixel + positions[1] - 1,
            collect(range(0.0, 0.31622776601683794; length=8)),
        )
        psi = marginal_angles[1].ψ
        h2 = sum(cis.(2 .* psi)) / length(psi)
        h4 = sum(cis.(4 .* psi)) / length(psi)
        h_matrix = ComplexF64[
            1.0              0.5*conj(h2)     0.5*h2
            0.5*h2           0.25             0.25*h4
            0.5*conj(h2)     0.25*conj(h4)    0.25
        ]
        reciprocal_condition = inv(cond(h_matrix))
        @test 1e-10 < reciprocal_condition < 1e-2

        marginal = _synthetic_ring_convolution(
            Float64,
            cp,
            geometry,
            positions,
            marginal_angles,
            stokes,
        )
        try
            default_stokes = ring_stokes(marginal, marginal_angles)
            diagnostic_stokes = ring_stokes(
                marginal,
                marginal_angles;
                rcond=1e-10,
            )
            @test all(isnan, default_stokes[:, :, :, 1])
            @test all(isfinite, diagnostic_stokes[:, :, :, 1])
        finally
            CUDA.unsafe_free!(marginal.accumulated)
        end
    end
end

"""
A real field requires real `m=0` coefficients: `a(l,-0) = a(l,0)` together with
`a(l,-m) = (-1)^m conj(a(l,m))` forces `a(l,0) = conj(a(l,0))`. The packed index
of `(l, 0)` is `l + 1`. Test data that skips this describes no real sky or beam,
so the convolved signal is not real either.
"""
function _realify_m0(coefficients, lmax)
    physical = copy(coefficients)
    for coefficient in 1:(lmax + 1), spin in 1:3
        physical[spin, coefficient] =
            complex(real(physical[spin, coefficient]), 0.0)
    end
    return physical
end

@testset "Ring convolution to map" begin
    cp = ConvolutionParams(
        nside_output=4,
        lstart=0,
        lstop=5,
        lmax=5,
        bnmax=2,
        κ=2,
        numofsky=2,
        numofbeam=2,
    )
    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    raw_alms = [
        _realify_m0(
            ComplexF64[
                complex(
                    sin(0.2 * spin + 0.3 * coefficient + sky),
                    cos(0.4 * spin - 0.1 * coefficient + 0.5 * sky),
                )
                for spin in 1:3, coefficient in 1:coefficient_count
            ],
            cp.lmax,
        )
        for sky in 1:cp.numofsky
    ]
    raw_blms = [
        _realify_m0(
            ComplexF64[
                complex(
                    sin(0.3 * spin + 0.2 * coefficient + beam),
                    cos(0.4 * spin - 0.1 * coefficient + 0.5 * beam),
                )
                for spin in 1:3, coefficient in 1:coefficient_count
            ],
            cp.lmax,
        )
        for beam in 1:cp.numofbeam
    ]

    if CUDA.functional()
        ring = 5
        geometry = ring_geometry(cp, ring)
        hit_positions = [2, 5, 9, 14]
        phi_tod, theta_tod, psi_tod = _ring_scan_tod(
            cp,
            geometry,
            hit_positions,
            9,
        )
        active, angles = make_ring_angles(
            cp,
            geometry,
            phi_tod,
            theta_tod,
            psi_tod,
        )
        beta_max = 1.05 * maximum(
            maximum(angle.β)
            for angle in angles
        )
        plan = ScanMomentPlan(
            kappa=cp.κ,
            nmax=calc_nmax(cp, cp.lstop),
            beta_order=10,
            beta_max=beta_max,
            precision=Float64,
        )
        table = build_wigner_chebyshev_table_cpu(cp, plan)
        device_table = upload_wigner_table_gpu(table)
        prepared_alm = prepare_alm(cp, raw_alms; precision=Float64)
        prepared_blm = prepare_blm(cp, raw_blms; precision=Float64)
        alm_gpu = upload_alm_gpu(prepared_alm)
        result = nothing
        try
            result = _convolve_ring_stages(
                cp, geometry, active, angles, alm_gpu, prepared_blm,
                device_table, plan; l_tile=3,
            )
            stokes = ring_stokes(result, angles)
            @test all(isfinite, stokes)
            @test !any(iszero, stokes)

            # Different skies and beams must give different maps.
            @test stokes[1, 1, :, :] != stokes[2, 1, :, :]
            @test stokes[1, 1, :, :] != stokes[1, 2, :, :]

            values = Array(result.accumulated)

            # A real sky convolved with a real beam must give a real signal.
            # <S> = accumulated[..., 1] is that signal averaged over hits, so
            # its imaginary part probes the spin (0,+2,-2) structure, the
            # negative-mode reconstruction, the beam conjugation and the
            # (1, 1/2, 1/2) weights together: a wrong conjugate or sign in any
            # of them leaves a non-zero imaginary part.
            @test maximum(abs, imag.(values[:, :, :, 1])) <
                1e-12 * maximum(abs, real.(values[:, :, :, 1]))

            # The recovered Stokes vector then reproduces the accumulator it
            # came from: rebuild <S> and <S e^{2i psi}> from (I, Q, U).
            for (active_index, angle) in pairs(angles),
                beam_index in 1:cp.numofbeam,
                sky_index in 1:cp.numofsky
                I = stokes[sky_index, beam_index, 1, active_index]
                Q = stokes[sky_index, beam_index, 2, active_index]
                U = stokes[sky_index, beam_index, 3, active_index]
                signal = [
                    _ideal_signal(I, Q, U, psi)
                    for psi in angle.ψ
                ]
                @test sum(signal) / angle.num ≈ real(
                    values[sky_index, beam_index, active_index, 1],
                ) atol=1e-9 rtol=1e-9
                @test sum(
                    signal[hit] * cis(2 * angle.ψ[hit])
                    for hit in 1:angle.num
                ) / angle.num ≈ values[
                    sky_index, beam_index, active_index, 2,
                ] atol=1e-9 rtol=1e-9
            end
        finally
            result === nothing || CUDA.unsafe_free!(result.accumulated)
            CUDA.unsafe_free!(active.positions)
            CUDA.unsafe_free!(alm_gpu.values)
            CUDA.unsafe_free!(device_table.coefficients)
            CUDA.unsafe_free!(device_table.pair_lookup)
        end
    end
end


const _LEGACY_INPUT_DIR = joinpath(@__DIR__, "data")

"""
Minimal reader for the legacy `.npy` inputs, so the test suite needs no extra
dependency. Both files are complex128 `(3, ncoeff)`; the beam happens to be
stored in Fortran order and the sky in C order.
"""
function _read_npy_complex(path)
    open(path) do io
        magic = read(io, 6)
        magic == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] ||
            error("$path is not a .npy file")
        read(io, 2)
        header_length = Int(ltoh(read(io, UInt16)))
        header = String(read(io, header_length))
        occursin("c16", header) ||
            error("$path is not complex128")
        shape = match(r"'shape':\s*\((\d+),\s*(\d+)\)", header)
        shape === nothing && error("$path has an unsupported shape")
        rows = parse(Int, shape[1])
        columns = parse(Int, shape[2])
        values = Vector{ComplexF64}(undef, rows * columns)
        read!(io, values)
        return occursin("'fortran_order': True", header) ?
            reshape(values, rows, columns) :
            permutedims(reshape(values, columns, rows), (2, 1))
    end
end

_legacy_packed_index(l, m, lmax) =
    m * (lmax + 1) - (m * (m - 1)) ÷ 2 + (l - m) + 1

@testset "Legacy CMB input" begin
    lmax = 47
    coefficient_count = (lmax + 1) * (lmax + 2) ÷ 2
    alm_raw = _read_npy_complex(
        joinpath(_LEGACY_INPUT_DIR, "alm_cmb_r0_nside16.npy"),
    )
    blm_raw = _read_npy_complex(
        joinpath(_LEGACY_INPUT_DIR, "blm_gauss_fwhm5_nside16.npy"),
    )
    @test size(alm_raw) == (3, coefficient_count)
    @test size(blm_raw) == (3, coefficient_count)

    # These describe real fields, so every m=0 coefficient is exactly real.
    # This is what the synthetic test data elsewhere in this file does not
    # satisfy without _realify_m0.
    m0_indices = [_legacy_packed_index(l, 0, lmax) for l in 0:lmax]
    @test all(iszero, imag.(alm_raw[:, m0_indices]))
    @test all(iszero, imag.(blm_raw[:, m0_indices]))

    # The Gaussian beam is axially symmetric: all T power sits at n=0 and all
    # E/B power at n=2, so bnmax=2 captures it exactly.
    function _fractional_power(coefficients, row, m)
        total = sum(
            abs2(coefficients[row, _legacy_packed_index(l, mm, lmax)])
            for mm in 0:lmax for l in mm:lmax
        )
        return sum(
            abs2(coefficients[row, _legacy_packed_index(l, m, lmax)])
            for l in m:lmax
        ) / total
    end
    @test _fractional_power(blm_raw, 1, 0) ≈ 1 atol=1e-20
    @test _fractional_power(blm_raw, 2, 2) ≈ 1 atol=1e-20
    @test _fractional_power(blm_raw, 3, 2) ≈ 1 atol=1e-20
    # Everything off the symmetric axis is numerical dust.
    @test maximum(_fractional_power(blm_raw, 1, m) for m in 1:5) < 1e-20
    @test maximum(
        _fractional_power(blm_raw, 2, m) for m in (0, 1, 3, 4, 5)
    ) < 1e-20
    @test maximum(
        _fractional_power(blm_raw, 3, m) for m in (0, 1, 3, 4, 5)
    ) < 1e-20

    if CUDA.functional()
        cp = ConvolutionParams(
            nside_output=16,
            lstart=0,
            lstop=12,
            lmax=lmax,
            bnmax=2,
            κ=2,
            numofsky=1,
            numofbeam=1,
        )
        ring = 16
        geometry = ring_geometry(cp, ring)
        @test geometry.pixel_count == 64
        hit_positions = [3, 11, 20, 34, 51]
        phi_tod, theta_tod, psi_tod = _ring_scan_tod(
            cp,
            geometry,
            hit_positions,
            8,
        )
        active, angles = make_ring_angles(
            cp,
            geometry,
            phi_tod,
            theta_tod,
            psi_tod,
        )
        beta_max = 1.05 * maximum(
            maximum(angle.β)
            for angle in angles
        )
        plan = ScanMomentPlan(
            kappa=cp.κ,
            nmax=calc_nmax(cp, cp.lstop),
            beta_order=12,
            beta_max=beta_max,
            precision=Float64,
        )
        table = build_wigner_chebyshev_table_cpu(cp, plan)
        device_table = upload_wigner_table_gpu(table)
        prepared_alm = prepare_alm(cp, alm_raw; precision=Float64)
        prepared_blm = prepare_blm(cp, blm_raw; precision=Float64)
        alm_gpu = upload_alm_gpu(prepared_alm)
        result = nothing
        try
            result = _convolve_ring_stages(
                cp, geometry, active, angles, alm_gpu, prepared_blm,
                device_table, plan; l_tile=4,
            )
            values = Array(result.accumulated)
            @test all(isfinite, values)

            # A real CMB sky convolved with a real beam gives a real signal.
            # Unlike the synthetic cases this uses unmodified production
            # inputs, so nothing about the reality condition is imposed by
            # the test itself.
            @test maximum(abs, imag.(values[:, :, :, 1])) <
                1e-12 * maximum(abs, real.(values[:, :, :, 1]))

            stokes = ring_stokes(result, angles)
            @test all(isfinite, stokes)
            @test !any(iszero, stokes)

            # A CMB sky is intensity dominated: r=0 gives no primordial B and
            # E is far below T, so the recovered I must dominate Q and U.
            recovered_I = abs.(stokes[1, 1, 1, :])
            recovered_Q = abs.(stokes[1, 1, 2, :])
            recovered_U = abs.(stokes[1, 1, 3, :])
            @test maximum(recovered_I) > 10 * maximum(recovered_Q)
            @test maximum(recovered_I) > 10 * maximum(recovered_U)
        finally
            result === nothing || CUDA.unsafe_free!(result.accumulated)
            CUDA.unsafe_free!(active.positions)
            CUDA.unsafe_free!(alm_gpu.values)
            CUDA.unsafe_free!(device_table.coefficients)
            CUDA.unsafe_free!(device_table.pair_lookup)
        end
    end
end


"""Re-pack harmonic coefficients for a smaller lmax; the packed index depends
on lmax, so the array cannot simply be truncated."""
function _retruncate_alm(coefficients, from_lmax, to_lmax)
    repacked = zeros(
        ComplexF64,
        3,
        (to_lmax + 1) * (to_lmax + 2) ÷ 2,
    )
    for l in 0:to_lmax, m in 0:l
        repacked[:, _legacy_packed_index(l, m, to_lmax)] =
            coefficients[:, _legacy_packed_index(l, m, from_lmax)]
    end
    return repacked
end

"""A scan that points exactly at the pixel centres of one ring, with psi spread
uniformly over [0, pi). The residual rotation is then exactly Rz(psi), so the
convolved signal is the beam-smoothed sky at the pixel centre with no averaging
over sub-pixel offsets, and h2 = h4 = 0 makes the binning solve exact."""
function _pixel_centre_ring_scan(cp, geometry, positions; hits=8)
    phi = Float64[]
    theta = Float64[]
    psi = Float64[]
    for position in positions, hit in 1:hits
        pixel = geometry.first_pixel + position - 1
        theta_pixel, phi_pixel = Scanvolver.pix2angRing(cp.resol, pixel)
        push!(theta, theta_pixel)
        push!(phi, phi_pixel)
        push!(psi, pi * (hit - 1) / hits)
    end
    return phi, theta, psi
end

function _convolve_pixel_centre_ring(cp, ring, raw_alm, raw_blm; l_tile=4)
    geometry = ring_geometry(cp, ring)
    positions = 1:geometry.pixel_count
    phi, theta, psi = _pixel_centre_ring_scan(cp, geometry, positions)
    active, angles = make_ring_angles(cp, geometry, phi, theta, psi)
    plan = ScanMomentPlan(
        kappa=cp.κ,
        nmax=calc_nmax(cp, cp.lstop),
        beta_order=8,
        beta_max=1e-3,
        precision=Float64,
    )
    device_table = upload_wigner_table_gpu(
        build_wigner_chebyshev_table_cpu(cp, plan),
    )
    alm_gpu = upload_alm_gpu(prepare_alm(cp, raw_alm; precision=Float64))
    prepared_blm = prepare_blm(cp, raw_blm; precision=Float64)
    result = _convolve_ring_stages(
        cp, geometry, active, angles, alm_gpu, prepared_blm, device_table, plan;
        l_tile=l_tile,
    )
    # Every pixel of the ring is hit, so the ring-local order is pixel order.
    stokes = ring_stokes(result, angles)
    pixels = copy(result.pixels)
    maximum_beta = maximum(maximum(angle.β) for angle in angles)
    CUDA.unsafe_free!(result.accumulated)
    CUDA.unsafe_free!(active.positions)
    CUDA.unsafe_free!(alm_gpu.values)
    CUDA.unsafe_free!(device_table.coefficients)
    CUDA.unsafe_free!(device_table.pair_lookup)
    return stokes, pixels, maximum_beta
end

"""Largest |m| > lstop Fourier amplitude of a ring, and the largest in-band one.
A convolution truncated at lstop can only produce in-band power, so this needs
no external reference at all."""
function _ring_out_of_band(values, lstop)
    count = length(values)
    out_of_band = 0.0
    in_band = 0.0
    for bin in 0:(count - 1)
        m = bin <= count ÷ 2 ? bin : bin - count
        amplitude = abs(sum(
            values[position] *
            cis(-2pi * (position - 1) * bin / count)
            for position in 1:count
        )) / count
        if abs(m) > lstop
            out_of_band = max(out_of_band, amplitude)
        else
            in_band = max(in_band, amplitude)
        end
    end
    return out_of_band, in_band
end

@testset "Convolution normalization" begin
    if CUDA.functional()
        lmax = 12
        cp = ConvolutionParams(
            nside_output=16,
            lstart=0,
            lstop=lmax,
            lmax=lmax,
            bnmax=2,
            κ=2,
            numofsky=1,
            numofbeam=1,
        )
        coefficient_count = (lmax + 1) * (lmax + 2) ÷ 2

        # (1) The per-l transfer function. With a pure spin-0 beam whose only
        # non-zero coefficient is b_T(l,0)=1, whatever l dependence survives is
        # the pipeline's own, and it must be the convolution-theorem factor
        # sqrt(4pi/(2l+1)) for an axially symmetric beam.
        unit_beam = zeros(ComplexF64, 3, coefficient_count)
        for l in 0:lmax
            unit_beam[1, _legacy_packed_index(l, 0, lmax)] = 1.0
        end
        for (mode_l, mode_m) in ((0, 0), (2, 1), (4, 1), (6, 3))
            sky = zeros(ComplexF64, 3, coefficient_count)
            value = mode_m == 0 ? ComplexF64(1.0) : ComplexF64(0.6, 0.8)
            sky[1, _legacy_packed_index(mode_l, mode_m, lmax)] = value

            stokes, pixels, maximum_beta =
                _convolve_pixel_centre_ring(cp, 16, sky, unit_beam)
            @test maximum_beta == 0.0

            alm_count = numberOfAlms(lmax, lmax)
            reference_alm = Alm(lmax, lmax, zeros(ComplexF64, alm_count))
            reference_alm.alm[almIndex(reference_alm, mode_l, mode_m)] = value
            reference = alm2map(reference_alm, cp.nside_output)

            transfer = stokes[1, 1, 1, :] ./
                reference.pixels[pixels]
            expected = sqrt(4pi / (2 * mode_l + 1))
            @test maximum(abs, transfer .- expected) < 1e-11 * expected
        end

        # (2) The real CMB sky and the axially symmetric Gaussian beam against
        # an independent spherical harmonic synthesis.
        sky = _retruncate_alm(
            _read_npy_complex(
                joinpath(_LEGACY_INPUT_DIR, "alm_cmb_r0_nside16.npy"),
            ),
            47,
            lmax,
        )
        beam = _retruncate_alm(
            _read_npy_complex(
                joinpath(_LEGACY_INPUT_DIR, "blm_gauss_fwhm5_nside16.npy"),
            ),
            47,
            lmax,
        )

        # Ring 16 is the case that used to fail: its colatitude makes the
        # residual R[3,3] land one ulp below 1.
        for ring in (15, 16, 24)
            stokes, pixels, maximum_beta =
                _convolve_pixel_centre_ring(cp, ring, sky, beam)
            @test maximum_beta == 0.0

            alm_count = numberOfAlms(lmax, lmax)
            smoothed_t = Alm(lmax, lmax, zeros(ComplexF64, alm_count))
            smoothed_e = Alm(lmax, lmax, zeros(ComplexF64, alm_count))
            smoothed_b = Alm(lmax, lmax, zeros(ComplexF64, alm_count))
            for l in 0:lmax, m in 0:l
                factor = sqrt(4pi / (2l + 1))
                window_t = factor * beam[1, _legacy_packed_index(l, 0, lmax)]
                window_p = factor * beam[2, _legacy_packed_index(l, 2, lmax)]
                index = almIndex(smoothed_t, l, m)
                packed = _legacy_packed_index(l, m, lmax)
                smoothed_t.alm[index] = sky[1, packed] * window_t
                smoothed_e.alm[index] = sky[2, packed] * window_p
                smoothed_b.alm[index] = sky[3, packed] * window_p
            end
            reference = alm2map(
                [smoothed_t, smoothed_e, smoothed_b],
                cp.nside_output,
            )

            recovered_i = stokes[1, 1, 1, :]
            recovered_q = stokes[1, 1, 2, :]
            recovered_u = stokes[1, 1, 3, :]
            reference_i = reference.i.pixels[pixels]
            # This beam has b_B = i*b_E, hence b_{+2} = 0 and b_{-2} = -2*b_E,
            # so using b_E as the polarized window is short by exactly -2.
            reference_q = -2 .* reference.q.pixels[pixels]
            reference_u = -2 .* reference.u.pixels[pixels]

            @test maximum(abs, recovered_i .- reference_i) <
                1e-12 * maximum(abs, reference_i)
            @test maximum(abs, recovered_q .- reference_q) <
                1e-9 * maximum(abs, reference_q)
            @test maximum(abs, recovered_u .- reference_u) <
                1e-9 * maximum(abs, reference_u)

            # (3) Band limit. Independent of any reference: a convolution
            # truncated at lstop cannot put power above |m| = lstop on a ring.
            for values in (recovered_i, recovered_q, recovered_u)
                out_of_band, in_band = _ring_out_of_band(values, lmax)
                @test out_of_band < 1e-12 * in_band
            end
        end
    end
end


@testset "Wigner Chebyshev CPU table" begin
    cp = ConvolutionParams(
        nside_output=2,
        lstart=1,
        lstop=4,
        lmax=4,
        bnmax=1,
        κ=1,
    )
    plan = ScanMomentPlan(
        kappa=cp.κ,
        nmax=calc_nmax(cp, cp.lstop),
        beta_order=16,
        beta_max=0.8,
        precision=Float64,
    )
    table = build_wigner_chebyshev_table_cpu(cp, plan)
    expected_pairs = Tuple{Int,Int}[
        (M, n)
        for n in -plan.nmax:plan.nmax
        for M in max(-2, n - plan.kappa):min(2, n + plan.kappa)
    ]

    @test collect(zip(Int.(table.pair_M), Int.(table.pair_n))) ==
        expected_pairs
    @test table.lstart == cp.lstart
    @test table.lstop == cp.lstop
    @test table.kappa == cp.κ
    @test table.nmax == plan.nmax
    @test table.beta_max == plan.beta_max
    @test size(table.coefficients) == (
        plan.beta_order,
        length(expected_pairs),
        cp.lstop - cp.lstart + 1,
    )
    @test eltype(table.coefficients) === Float64
    @test all(eachindex(table.first_l)) do pair_index
        table.first_l[pair_index] == max(
            cp.lstart,
            abs(table.pair_M[pair_index]),
            abs(table.pair_n[pair_index]),
        )
    end

    beta = 0.37
    basis = zeros(Float64, plan.beta_order)
    Scanvolver._chebyshev_values!(basis, plan, beta)
    pair_index = findfirst(eachindex(table.pair_M)) do index
        table.pair_M[index] == 1 && table.pair_n[index] == 0
    end
    l_index = 1 - table.lstart + 1
    approximation = sum(
        @view(table.coefficients[:, pair_index, l_index]) .* basis,
    )
    expected = -sin(beta) / sqrt(2)
    @test approximation ≈ expected atol=3e-12 rtol=3e-12

    plan32 = ScanMomentPlan(
        kappa=cp.κ,
        nmax=calc_nmax(cp, cp.lstop),
        beta_order=8,
        beta_max=0.8,
        precision=Float32,
    )
    @test eltype(
        build_wigner_chebyshev_table_cpu(cp, plan32).coefficients,
    ) === Float32

    bad_kappa = ScanMomentPlan(
        kappa=2,
        nmax=1,
        beta_order=4,
        beta_max=0.8,
    )
    @test_throws ArgumentError build_wigner_chebyshev_table_cpu(
        cp,
        bad_kappa,
    )
    bad_nmax = ScanMomentPlan(
        kappa=1,
        nmax=0,
        beta_order=4,
        beta_max=0.8,
    )
    @test_throws ArgumentError build_wigner_chebyshev_table_cpu(
        cp,
        bad_nmax,
    )
    hwp_cp = ConvolutionParams(
        nside_output=2,
        lstop=4,
        lmax=4,
        bnmax=1,
        κ=1,
        HWP=true,
        HWP_rpm=60,
    )
    @test_throws ArgumentError build_wigner_chebyshev_table_cpu(
        hwp_cp,
        plan,
    )
end

@testset "ringD GPU" begin
    cp = ConvolutionParams(
        nside_output=2,
        lstop=4,
        lmax=4,
        bnmax=0,
        κ=4,
    )

    if CUDA.functional()
        beta = 0.731
        ringd_device = CUDA.@sync _ringd_gpu(
            cp,
            beta,
        )
        ringd = Array(ringd_device)
        @test size(ringd) == (9, 9, 5)
        @test eltype(ringd_device) == ComplexF64
        @test all(iszero, imag.(ringd))

        cosine_beta = cos(beta)
        sine_beta = sin(beta)
        expected_l1 = [
            (1 + cosine_beta) / 2 sine_beta / sqrt(2) (1 - cosine_beta) / 2
            -sine_beta / sqrt(2) cosine_beta sine_beta / sqrt(2)
            (1 - cosine_beta) / 2 -sine_beta / sqrt(2) (1 + cosine_beta) / 2
        ]
        @test real.(ringd[1:3, 1:3, 2]) ≈
            expected_l1 atol=2e-12 rtol=2e-12

        for l in 0:cp.lstop
            count = 2l + 1
            block = @view ringd[1:count, 1:count, l + 1]
            for first_column in 1:count
                for second_column in 1:count
                    inner_product = zero(ComplexF64)
                    for row in 1:count
                        inner_product +=
                            conj(block[row, first_column]) *
                            block[row, second_column]
                    end
                    expected =
                        first_column == second_column ? 1.0 : 0.0
                    @test inner_product ≈
                        expected atol=3e-12 rtol=3e-12
                end
            end
        end

        identity_ringd_device = CUDA.@sync _ringd_gpu(cp, 0.0)
        identity_ringd = Array(identity_ringd_device)
        for l in 0:cp.lstop
            count = 2l + 1
            block = @view identity_ringd[
                1:count,
                1:count,
                l + 1,
            ]
            for row in 1:count, column in 1:count
                expected = row == column ? 1.0 : 0.0
                @test real(block[row, column]) ≈
                    expected atol=2e-12 rtol=2e-12
            end
        end

        ringd32_device = CUDA.@sync _ringd_gpu(
            cp,
            beta;
            precision=Float32,
        )
        @test eltype(ringd32_device) == ComplexF32
        @test Array(ringd32_device) ≈
            ringd atol=8e-5 rtol=8e-5
        @test_throws ArgumentError _ringd_gpu(
            cp,
            beta;
            precision=Float16,
        )
    else
        @test_throws ArgumentError _ringd_gpu(
            cp,
            0.731,
        )
    end
end

@testset "Wigner Chebyshev table against ringD and upload" begin
    cp = ConvolutionParams(
        nside_output=2,
        lstart=1,
        lstop=4,
        lmax=4,
        bnmax=1,
        κ=1,
    )
    plan = ScanMomentPlan(
        kappa=cp.κ,
        nmax=calc_nmax(cp, cp.lstop),
        beta_order=16,
        beta_max=0.8,
        precision=Float64,
    )
    table = build_wigner_chebyshev_table_cpu(cp, plan)

    if CUDA.functional()
        # Every coefficient of the CPU table, evaluated at one beta, against
        # the GPU ringD recursion: two independent recurrence codes.
        beta = 0.37
        chebyshev = zeros(Float64, plan.beta_order)
        Scanvolver._chebyshev_values!(chebyshev, plan, beta)
        context = ringd_context(cp; precision=Float64)
        ringd = Array(CUDA.@sync ringd_blocks_recursion_gpu!(context, cp, beta))
        free_ringd_context!(context)
        Mmax_global = calc_Mmax(cp, cp.lstop)
        for pair_index in eachindex(table.pair_M)
            M = Int(table.pair_M[pair_index])
            n = Int(table.pair_n[pair_index])
            for l in cp.lstart:cp.lstop
                l_index = l - cp.lstart + 1
                approximation = sum(
                    @view(table.coefficients[:, pair_index, l_index]) .*
                    chebyshev,
                )
                if l < table.first_l[pair_index]
                    @test approximation == 0.0
                else
                    Mmax_l = min(Mmax_global, l)
                    reference = real(ringd[
                        M + l + 1,
                        n + Mmax_l + 1,
                        l_index,
                    ])
                    @test approximation ≈
                        reference atol=3e-12 rtol=3e-12
                end
            end
        end

        device_table = upload_wigner_table_gpu(table)
        @test size(device_table.coefficients) == size(table.coefficients)
        @test device_table.beta_order == plan.beta_order
        @test device_table.Mmax == calc_Mmax(cp, cp.lstop)
        pair_lookup = Array(device_table.pair_lookup)
        for pair_index in eachindex(table.pair_M)
            M = Int(table.pair_M[pair_index])
            n = Int(table.pair_n[pair_index])
            @test pair_lookup[
                M + Int(device_table.Mmax) + 1,
                n + table.nmax + 1,
            ] == pair_index
        end

        plan32 = ScanMomentPlan(
            kappa=cp.κ,
            nmax=calc_nmax(cp, cp.lstop),
            beta_order=8,
            beta_max=0.8,
            precision=Float32,
        )
        table32 = build_wigner_chebyshev_table_cpu(cp, plan32)
        @test eltype(upload_wigner_table_gpu(table32).coefficients) === Float32
    end
end

@testset "Euler ZYZ degeneracy" begin
    wrap = Scanvolver.wrap_pm_pi
    rebuild(α, β, γ) = Matrix(Scanvolver.Rzyz(α, β, γ))

    # The defining property: the extracted angles must rebuild the rotation.
    # This is what a degenerate beta used to break.
    for (a, b, c) in (
        (0.3, 1e-3, 0.7),
        (-1.2, 0.4, 2.0),
        (2.9, 1.5, -2.7),
        (0.1, pi - 1e-3, -0.5),
        (-0.8, 1e-9, 1.1),
    )
        R = Matrix(Scanvolver.Rzyz(a, b, c))
        α, β, γ = Scanvolver.euler_zyz_from_R(R)
        @test rebuild(α, β, γ) ≈ R atol=1e-14
        @test β ≈ b atol=1e-14
    end

    # beta = 0 exactly: alpha and gamma are individually undefined and only
    # their sum is determined, so the convention is alpha = 0, gamma = psi.
    for ψ in (0.0, 0.3, 0.7853981633974483, 1.9634954084936207, -2.5)
        R = Matrix(Scanvolver.Rzyz(0.0, 0.0, ψ))
        α, β, γ = Scanvolver.euler_zyz_from_R(R)
        @test α == 0.0
        @test β == 0.0
        @test wrap(α + γ) ≈ wrap(ψ) atol=1e-15
        @test rebuild(α, β, γ) ≈ R atol=1e-15
    end

    # Regression: a residual rotation that is the identity only up to roundoff.
    # acos maps a one-ulp deficit in R[3,3] to beta = 1.49e-8, which clears any
    # sensible threshold, so the old sin(beta) test sent this to the general
    # branch and read alpha off two entries that are pure roundoff.
    @test acos(prevfloat(1.0)) > 1e-8
    for ψ in (0.0, 0.7853981633974483, 1.9634954084936207)
        R = Matrix(Scanvolver.Rzyz(0.0, 0.0, ψ))
        R[3, 3] = prevfloat(1.0)
        R[1, 3] = -4.643e-17
        R[2, 3] = 0.0
        R[3, 1] = 2.3e-17
        R[3, 2] = -1.1e-17
        α, β, γ = Scanvolver.euler_zyz_from_R(R)
        @test α == 0.0
        @test β < 1e-15
        @test wrap(α + γ) ≈ wrap(ψ) atol=1e-15
    end

    # beta = pi.
    for (a, c) in ((0.0, 0.4), (1.1, -0.9))
        R = Matrix(Scanvolver.Rzyz(a, pi, c))
        α, β, γ = Scanvolver.euler_zyz_from_R(R)
        @test β ≈ pi atol=1e-14
        @test rebuild(α, β, γ) ≈ R atol=1e-14
    end

    # split_rotation with the pointing exactly on the pixel centre. The residual
    # is Rz(psi), so alpha + gamma must be psi for every pixel of the ring; this
    # used to hold on some rings and fail on others depending on the last bit of
    # the colatitude.
    cp = ConvolutionParams(nside_output=16, lstop=12, lmax=12, bnmax=2, κ=2)
    for ring in (2, 11, 15, 16, 19, 24)
        geometry = ring_geometry(cp, ring)
        for position in 1:geometry.pixel_count
            pixel = geometry.first_pixel + position - 1
            θ_pixel, φ_pixel = Scanvolver.pix2angRing(cp.resol, pixel)
            for ψ in (0.0, 0.7853981633974483, 1.9634954084936207)
                α, β, γ = Scanvolver.split_rotation(
                    φ_pixel,
                    θ_pixel,
                    0.0,
                    0.0,
                    ψ,
                )
                @test β < 1e-15
                @test wrap(α + γ) ≈ wrap(ψ) atol=1e-14
            end
        end
    end
end


@testset "Angles" begin
    φ = [0.0, 0.0, 0.0]
    θ = [0.0, 0.0, 0.0]
    ψ = [0.0, 0.0, 0.0]
    times = [0.0, 0.25, 0.5]

    cp = ConvolutionParams(
        nside_output=1,
        HWP=true,
        HWP_rpm=60,
    )
    angles = Angles(
        cp;
        pix=1,
        φ=φ,
        θ=θ,
        ψ=ψ,
        time_array=times,
    )
    @test angles.num == 3
    @test angles.time_array == times
    @test angles.η ≈ [0.0, π / 2, π]
    @test length(angles.α) == 3
    @test length(angles.β) == 3
    @test length(angles.γ) == 3

    cp_without_hwp = ConvolutionParams(nside_output=1)
    angles_without_time = Angles(
        cp_without_hwp;
        pix=1,
        φ=φ,
        θ=θ,
        ψ=ψ,
    )
    @test isempty(angles_without_time.time_array)
    @test isempty(angles_without_time.η)

    # Angles must rotate from the output-pixel frame to each sample pointing.
    # Reversing this residual leaves pixel-centre samples unchanged but flips
    # the leading off-centre leakage terms.
    cp_off_center = ConvolutionParams(nside_output=16)
    pixel = 1537
    θ_pixel, φ_pixel = Scanvolver.pix2angRing(cp_off_center.resol, pixel)
    φ_off_center = φ_pixel .+ [1.0e-4, -1.5e-4, 0.5e-4]
    θ_off_center = θ_pixel .+ [-0.8e-4, 0.4e-4, 1.2e-4]
    ψ_off_center = [0.1, -0.7, 1.2]
    angles_off_center = Angles(
        cp_off_center;
        pix=pixel,
        φ=φ_off_center,
        θ=θ_off_center,
        ψ=ψ_off_center,
    )
    for sample_index in eachindex(ψ_off_center)
        reconstructed =
            Scanvolver.Rzyz(φ_pixel, θ_pixel, 0.0) *
            Scanvolver.Rzyz(
                angles_off_center.α[sample_index],
                angles_off_center.β[sample_index],
                angles_off_center.γ[sample_index],
            )
        expected = Scanvolver.Rzyz(
            φ_off_center[sample_index],
            θ_off_center[sample_index],
            ψ_off_center[sample_index],
        )
        # Equivalent Euler rotations can differ slightly across libm and
        # processor implementations near this off-centre configuration.
        @test reconstructed ≈ expected atol=2e-13
    end

    @test_throws ArgumentError Angles(
        cp;
        pix=1,
        φ=φ,
        θ=θ,
        ψ=ψ,
    )
    @test_throws ArgumentError Angles(
        cp;
        pix=1,
        φ=φ,
        θ=θ,
        ψ=ψ,
        time_array=times[1:2],
    )
    @test_throws ArgumentError Angles(
        cp_without_hwp;
        pix=1,
        φ=φ,
        θ=θ[1:2],
        ψ=ψ,
    )
end

include("test_hdf5_pipelines.jl")
