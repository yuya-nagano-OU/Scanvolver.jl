using Test
using Scanvolver
import LinearAlgebra

"""Independent small-matrix exponential oracle, not the Wigner recurrence."""
function _kappa_matrix_reference(l, beta)
    generator = zeros(2l + 1, 2l + 1)
    for (index, m) in enumerate(-l:(l - 1))
        coupling = 0.5sqrt((l - m) * (l + m + 1))
        generator[index + 1, index] = -coupling
        generator[index, index + 1] = coupling
    end
    return exp(beta * generator)
end

@testset "CPU kappa estimator" begin
    @testset "Independent rotation and complete omitted band" begin
        cases = ((0, 0, 0.0), (1, 0, 0.2), (1, 1, 0.4), (2, 1, 0.7),
                 (4, 2, 0.1), (5, 5, 0.45), (8, 3, 0.2), (8, 8, 0.6))
        for (l, nmax, beta) in cases
            direct = _kappa_matrix_reference(l, beta)
            @test direct' * direct ≈ Matrix{Float64}(LinearAlgebra.I, 2l + 1, 2l + 1) atol=2e-14
            for normalization in (:diagonal, :absolute), threshold in (1e-2, 1e-6)
                profile = zeros(l + nmax + 1)
                for n in -nmax:nmax, M in -l:l
                    denominator = normalization === :diagonal ? abs(direct[n + l + 1, n + l + 1]) : 1.0
                    q = abs(M - n)
                    profile[q + 1] = max(profile[q + 1], abs(direct[M + l + 1, n + l + 1]) / denominator)
                end
                expected = 0
                for q in 1:(l + nmax)
                    profile[q + 1] > threshold && (expected = q)
                end
                report = estimate_kappa(; nside=4, lmax=l, nmax=nmax, ε=threshold,
                                        beta_max=beta, normalization, details=true)
                @test report.kappa == expected
                @test report.scope === :lmax_at_beta_max
                @test report.ratios ≈ profile[1:length(report.ratios)] rtol=2e-10 atol=2e-14
                @test report.omitted_maximum_bound <= threshold * (1 + 1e-12)
                @test estimate_kappa(; nside=4, lmax=l, nmax=nmax, ε=threshold,
                                     beta_max=beta, normalization) == expected
                if report.tail_start <= l + nmax
                    @test maximum(profile[(report.tail_start + 1):end]) <= report.tail_bound * (1 + 1e-10) + 2e-14
                end
            end
        end
    end

    @testset "Known normalization and nonmonotone profiles" begin
        # A one-step element is sin(beta)/sqrt(2), while d^1_00=cos(beta).
        beta = 0.2
        info = estimate_kappa(; nside=1, lmax=1, nmax=0, ε=1e-8, beta_max=beta, details=true)
        @test info.ratios[2] ≈ abs(tan(beta)) / sqrt(2) rtol=5e-14 atol=5e-15
        @test info.kappa == 1
        @test estimate_kappa(; nside=1, lmax=1, nmax=0, ε=0.2, beta_max=beta) == 0

        # At pi the nearest off-diagonal vanishes, but the next is unity.
        # Stopping at the first sub-threshold entry would incorrectly return 0.
        info = estimate_kappa(; nside=1, lmax=1, nmax=1, ε=1e-2,
                              beta_max=pi, normalization=:absolute, details=true)
        @test info.ratios[2] < 1e-14
        @test info.ratios[3] ≈ 1.0 atol=2e-14
        @test info.kappa == 2

        @test estimate_kappa(; nside=1, lmax=0, nmax=0, ε=1e-12) == 0
        @test estimate_kappa(; nside=16, lmax=30, nmax=30, ε=1e-12, beta_max=0.0) == 0
        @test_throws DomainError estimate_kappa(; nside=1, lmax=1, nmax=0, ε=1e-2, beta_max=pi/2)
        @test estimate_kappa(; nside=1, lmax=1, nmax=0, ε=1e-2,
                             beta_max=pi/2, normalization=:absolute) == 1
    end

    @testset "ell1000, nmax200, nside512 regression" begin
        report = estimate_kappa(; nside=512, lmax=1000, nmax=200, ε=1e-2, details=true)
        @test report.kappa == 5
        @test report.beta_max == automatic_beta_max(512; precision=Float64)
        @test report.minimum_diagonal ≈ 0.17337762929226364 rtol=5e-7
        # Independent exp(-i beta J_y) reference computed on Penzias in Float64.
        @test report.ratios[6] ≈ 0.0495986117895856 rtol=5e-7
        @test report.ratios[7] ≈ 0.008865203352630708 rtol=5e-7
        @test report.ratios[8] ≈ 0.0013488937993411873 rtol=5e-7
        @test report.omitted_maximum_bound <= 1e-2
        @test report.tail_start < 30  # Far-away components were bounded, not materialized.
        @test estimate_kappa(nside=512, lmax=1000, nmax=200, ε=1e-2) == 5
        @test estimate_kappa(; nside=512, lmax=1000, nmax=200, ε=1e-2, beta_max=2.13/1000) == 6
        @test estimate_kappa(; nside=512, lmax=1000, nmax=200, ε=1e-2, normalization=:absolute) == 4
        @test estimate_kappa(; nside=512, lmax=1000, nmax=30, ε=1e-12) == 15
    end

    @testset "Required geometry keywords and optional epsilon" begin
        @test estimate_kappa(; nside=512, lmax=1000, nmax=200, ε=1e-8) == 12
        # Geometry keywords may be reordered; epsilon alone can be positional.
        @test estimate_kappa(; ε=1e-8, nmax=200, lmax=1000, nside=512) == 12
        info = estimate_kappa(; nside=512, lmax=1000, nmax=200, ε=1e-8, details=true)
        @test info.threshold == 1e-8  # Preserve the existing diagnostics field.
        @test estimate_kappa(1e-8; nside=512, lmax=1000, nmax=200) == 12
        @test estimate_kappa(1e-5; nside=512, lmax=1000, nmax=200) == 9
        @test estimate_kappa(nside=512, lmax=1000, nmax=200) == 9
        default_info = estimate_kappa(; nside=512, lmax=1000, nmax=200, details=true)
        @test default_info.threshold == 1e-5
        @test default_info.kappa == 9
        @test estimate_kappa(1e-2; nside=512, lmax=1000, nmax=200, normalization=:absolute) == 4
        positional_info = estimate_kappa(1e-12; nside=16, lmax=30, nmax=30,
                                         beta_max=0.0, diagonal_atol=0.0, details=true)
        @test positional_info.kappa == 0
        @test positional_info.threshold == 1e-12
        @test_throws MethodError estimate_kappa(1000, 200, 512, 1e-8)
        @test_throws MethodError estimate_kappa(512, 1000, 200, 1e-8)
        @test_throws MethodError estimate_kappa(1000, 200, 512; threshold=1e-8)
        @test_throws UndefKeywordError estimate_kappa(512; lmax=1000, nmax=200)
        @test_throws MethodError estimate_kappa(; nside=512, lmax=1000, nmax=200,
                                               ε=1e-8, threshold=1e-8)
        @test_throws ArgumentError estimate_kappa(1e-8; nside=512, lmax=1000, nmax=200, ε=1e-5)
        @test_throws MethodError estimate_kappa("1e-5"; nside=512, lmax=1000, nmax=200)
        @test_throws UndefKeywordError estimate_kappa()
        @test_throws UndefKeywordError estimate_kappa(; lmax=1000, nmax=200, ε=1e-8)
        @test_throws UndefKeywordError estimate_kappa(; nside=512, nmax=200, ε=1e-8)
        @test_throws UndefKeywordError estimate_kappa(; nside=512, lmax=1000, ε=1e-8)
        @test_throws UndefKeywordError estimate_kappa(1e-8; lmax=1000, nmax=200)
        @test_throws UndefKeywordError estimate_kappa(1e-8; nside=512, nmax=200)
        @test_throws UndefKeywordError estimate_kappa(1e-8; nside=512, lmax=1000)
    end

    @testset "Input validation" begin
        @test_throws ArgumentError estimate_kappa(; nside=1, lmax=-1, nmax=0, ε=1e-2)
        @test_throws ArgumentError estimate_kappa(; nside=1, lmax=4, nmax=-1, ε=1e-2)
        @test_throws ArgumentError estimate_kappa(; nside=1, lmax=4, nmax=5, ε=1e-2)
        @test_throws ArgumentError estimate_kappa(; nside=0, lmax=4, nmax=2, ε=1e-2)
        for threshold in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError estimate_kappa(; nside=1, lmax=4, nmax=2, ε=threshold)
            @test_throws ArgumentError estimate_kappa(threshold; nside=1, lmax=4, nmax=2)
        end
        for beta in (-0.1, 4.0, Inf, NaN)
            @test_throws ArgumentError estimate_kappa(; nside=1, lmax=4, nmax=2, ε=1e-2, beta_max=beta)
        end
        for floor in (-0.1, Inf, NaN)
            @test_throws ArgumentError estimate_kappa(; nside=1, lmax=4, nmax=2, ε=1e-2, diagonal_atol=floor)
        end
        @test_throws ArgumentError estimate_kappa(; nside=1, lmax=4, nmax=2, ε=1e-2, normalization=:peak)
        # Near-zero diagonals are never replaced with a floor silently.
        @test_throws DomainError estimate_kappa(; nside=1, lmax=1, nmax=0, ε=1e-2, diagonal_atol=1.0)
    end
end
