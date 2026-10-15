using Test
using CUDA
using Scanvolver
using HDF5
using HDF5: attributes
using Healpix

include(joinpath(@__DIR__, "..", "detector_grouped_pipeline.jl"))
using .DetectorGroupedHDF5

const OFFSETS = (0.0, pi / 2, pi / 4, 3pi / 4)
const WEIGHTS = ones(4)

@testset "four-detector sufficient statistics and Scanvolver U sign" begin
    intensity, q, u = 2.3, -0.7, 0.41
    base_psi = 0.37
    angles = [(; num=1, ψ=[base_psi])]
    for T in (Float32, Float64)
        tolerance = T === Float32 ? 5e-7 : 2e-14
        detector_values = zeros(Complex{T}, 1, 4, 1, 2)
        for detector in 1:4
            psi = base_psi + OFFSETS[detector]
            signal = intensity + q*cos(2psi) + u*sin(2psi)
            detector_values[1, detector, 1, 1] = signal
            detector_values[1, detector, 1, 2] = signal * cis(2base_psi)
        end
        grouped = reduce_detector_accumulators(detector_values, OFFSETS, WEIGHTS)
        expected_spin0 = intensity
        expected_spin2 = (q + im*u) / 2
        @test grouped[1, 1, 1, 1] ≈ expected_spin0 atol=tolerance
        @test grouped[1, 1, 1, 2] ≈ expected_spin2 atol=tolerance

        stokes = grouped_ring_stokes(grouped, angles, OFFSETS, WEIGHTS;
                                      rcond=1e-10, precision=T)
        @test stokes[1, 1, :, 1] ≈ T.([intensity, q, u]) atol=tolerance

        # Omitting the detector phase is not an equivalent operation.
        wrong_grouped = zeros(Complex{T}, 1, 1, 1, 2)
        wrong_grouped[1,1,1,1] = sum(detector_values[1,:,1,1]) / 4
        wrong_grouped[1,1,1,2] = sum(detector_values[1,:,1,2]) / 4
        wrong_stokes = grouped_ring_stokes(wrong_grouped, angles, OFFSETS, WEIGHTS;
                                            rcond=1e-10, precision=T)
        @test !isapprox(wrong_stokes[1,1,2,1], q; atol=1e-3)
        @test !isapprox(wrong_stokes[1,1,3,1], u; atol=1e-3)
    end
end

@testset "group validation" begin
    values = zeros(ComplexF32, 2, 8, 3, 2)
    grouped = reduce_detector_accumulators(values, OFFSETS, WEIGHTS)
    @test size(grouped) == (2, 2, 3, 2)
    @test_throws DimensionMismatch reduce_detector_accumulators(
        zeros(ComplexF32, 1, 5, 1, 2), OFFSETS, WEIGHTS,
    )
    @test_throws ArgumentError reduce_detector_accumulators(values, OFFSETS, [1, 1, 0, 1])
end

@testset "small HDF5 integration" begin
    if !CUDA.functional()
        @test_skip "CUDA is unavailable"
    else
        cp = ConvolutionParams(nside_output=2, lmax=3, lstop=3, bnmax=2,
                               κ=1, numofsky=1, numofbeam=4)
        coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
        raw_alm = [zeros(ComplexF64, 3, coefficient_count)]
        raw_alm[1][1,1] = 1
        raw_blm = [zeros(ComplexF64, 3, coefficient_count) for _ in 1:4]
        for beam in raw_blm
            beam[1,1] = 1
        end
        alm = prepare_alm(cp, raw_alm; precision=Float32)
        blm = prepare_blm(cp, raw_blm; precision=Float32)
        geometry = ring_geometry(cp, 2)
        phi = fill(geometry.phi_pixels[1], 2)
        theta = fill(geometry.theta, 2)
        psi = [0.1, 0.7]
        buffer = build_ring_scan_buffer(cp, phi, theta, psi)
        output_directory = mktempdir()
        output_path = joinpath(output_directory, "grouped.h5")
        result = convolve_detector_groups_to_hdf5(
            cp, buffer, alm, blm; output_path, devices=[0], rings=[2],
            group_batch_size=1, l_tile=2, map_precision=Float32,
            rcond=1e-10, reclaim_interval=0,
            sky_ids=["sky-1"], input_beam_ids=["Qp", "Qm", "Up", "Um"],
            group_ids=["orientation-1"],
        )
        @test result == output_path
        h5open(output_path, "r") do file
            @test read(attributes(file)["schema"]) == "EBCoDeCMB.detector_grouped_hdf5.v1"
            @test read(attributes(file)["polarization_convention"]) == "x2=Q+iU"
            @test read(attributes(file)["input_numofbeam"]) == 4
            @test read(attributes(file)["numofdetector_groups"]) == 1
            @test size(file["map"]) == (cp.resol.numOfPixels, 3, 1, 1)
            @test read(file["rings/completed"])[2] == 1
            @test read(file["detector_labels"]) == ["Qp", "Qm", "Up", "Um"]
        end
        ring_map = read_group_ring_hdf5(output_path, 2)
        @test size(ring_map) == (1, 1, 3, geometry.pixel_count)
        @test any(isfinite, ring_map)
    end
end
