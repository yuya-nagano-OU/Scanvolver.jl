# Both HDF5 pipelines. convolve_to_hdf5 is compared with maps built from the
# CPU stage references (cpu_reference.jl); the detector-grouped layout is
# checked against a hand-computed quartet and on a small HDF5 run.
using HDF5: HDF5, h5open, attributes
using Healpix: ang2pixRing, boundariesRing
using LinearAlgebra: norm
using SHA: sha256

# Declared before evaluating any new-pipeline result. These are pointwise
# tolerances on normalized test signals, not fitted to measured differences.
const _PIPELINE_TOLERANCES = Dict(Float32 => (atol=2e-6, rtol=2e-5),
                                  Float64 => (atol=1e-12, rtol=1e-11))

const _DETECTOR_TEST_OFFSETS = (0.0, pi / 2, pi / 4, 3pi / 4)
const _DETECTOR_TEST_WEIGHTS = ones(4)

function _pipeline_fixture(; nside=8, lmax=12, bnmax=4, kappa=3)
    cp = ConvolutionParams(nside_output=nside, lmax=lmax, lstop=lmax,
                           bnmax=bnmax, κ=kappa, numofsky=2, numofbeam=5)
    count = (lmax+1)*(lmax+2) ÷ 2
    function coefficients(realization, beam)
        out = zeros(ComplexF64, 3, count)
        for m in 0:lmax, l in m:lmax, spin in 1:3
            spin > 1 && l < 2 && continue
            index = m*(2lmax+1-m) ÷ 2 + l + 1
            phase = 0.71realization + 0.27l + 0.43m + 0.19spin
            value = complex(sin(phase), m == 0 ? 0.0 : 0.8cos(1.3phase))
            out[spin,index] = value / (l+1)^2 *
                (beam ? exp(-0.008l*(l+1)) * cis(0.17realization*m) : 1)
        end
        return out
    end
    alms = [coefficients(s, false) for s in 1:cp.numofsky]
    blms = [coefficients(b+3, true) for b in 1:cp.numofbeam]
    phi, theta, psi = Float64[], Float64[], Float64[]
    scanned = unique(sort([1,2,nside-1,nside,nside+1,2nside-1,2nside,2nside+1,
                          3nside,3nside+1,4nside-2,4nside-1]))
    rejected_pixel = 0
    for ring in scanned
        geometry = ring_geometry(cp, ring)
        positions = iseven(ring) ? (1:geometry.pixel_count) : (1:2:geometry.pixel_count)
        for position in positions
            pixel = geometry.first_pixel + position - 1
            singular = ring == nside && position == 1
            singular && (rejected_pixel = pixel)
            for hit in 0:7
                amplitude = 0.22/nside
                target_theta = geometry.theta + amplitude*sin(2pi*hit/8)
                target_phi = geometry.phi_pixels[position] + amplitude*cos(2pi*hit/8)/sin(geometry.theta)
                # Keep each perturbed pointing inside its intended pixel.
                for attempt in 1:12
                    ang2pixRing(cp.resol, target_theta, mod2pi(target_phi)) == pixel && break
                    target_theta = (target_theta + geometry.theta)/2
                    target_phi = (target_phi + geometry.phi_pixels[position])/2
                end
                @assert ang2pixRing(cp.resol, target_theta, mod2pi(target_phi)) == pixel
                push!(phi, mod2pi(target_phi)); push!(theta, target_theta)
                push!(psi, singular ? 0.0 : 2pi*hit/8)
            end
        end
    end
    return cp, alms, blms, build_ring_scan_buffer(cp, phi, theta, psi), rejected_pixel
end

# The reference map: every ring through the CPU reference of each stage (scan
# moments, sky ring, effective beam, contraction), then the per-pixel solve the
# pipeline also uses. Shared with the pipeline: the CPU-built Wigner table, the
# GPU ringD recursion and ring_stokes, each tested on its own in test/runtests.jl.
function _pipeline_reference(cp, buffer, alm, blm, T; tile=4)
    CUDA.device!(0)
    plan = automatic_scan_plan(cp; precision=T)
    table = build_wigner_chebyshev_table_cpu(cp, plan)
    reference = fill(T(NaN), cp.numofsky, cp.numofbeam, 3, cp.resol.numOfPixels)
    context = ringd_context(cp; precision=T)
    try
        for ring in buffer.first_ring:buffer.last_ring
            geometry = ring_geometry(cp, ring)
            active, angles = make_ring_angles(cp, geometry, buffer)
            active === nothing && continue
            positions = Int.(Array(active.positions))
            CUDA.unsafe_free!(active.positions)
            ringd = Array(ringd_blocks_recursion_gpu!(context, cp, geometry.theta))
            accumulated = _ring_convolution_cpu_reference(
                cp, geometry, positions, angles, table, alm, blm, ringd, plan, tile)
            pixels = geometry.first_pixel .+ positions .- 1
            result = RingConvolution{T}(ring, geometry.first_pixel, geometry.pixel_count,
                                        positions, pixels, CuArray(accumulated))
            reference[:, :, :, pixels] = ring_stokes(result, angles; rcond=1e-2, precision=T)
            CUDA.unsafe_free!(result.accumulated)
        end
    finally
        free_ringd_context!(context)
    end
    return reference
end

function _compare_pipeline_maps(reference, actual, T)
    tol = _PIPELINE_TOLERANCES[T]
    @test size(reference) == size(actual)
    @test isnan.(reference) == isnan.(actual)
    @test isfinite.(reference) == isfinite.(actual)
    for stokes in 1:3
        ref = Float64.(vec(reference[:,:,stokes,:]))
        got = Float64.(vec(actual[:,:,stokes,:]))
        valid = isfinite.(ref)
        @test count(valid) > 0
        delta = abs.(got[valid] .- ref[valid])
        @test all(delta .<= tol.atol .+ tol.rtol .* abs.(ref[valid]))
    end
    return nothing
end

@testset "nside-derived beta envelope" begin
    @test automatic_beta_max(256; precision=Float64) > 0.004172560737300081
    @test automatic_beta_max(512; precision=Float64) > 0.002087055235497685
    @test automatic_beta_max(256) < 0.004173
    for nside in (1,2,8,256,512), T in (Float32,Float64)
        cp = ConvolutionParams(nside_output=nside, lmax=2, lstop=2, bnmax=2)
        bound = automatic_beta_max(nside; precision=T)
        pixels = unique(vcat([1,cp.resol.numOfPixels],
            [first(ring_pixel_range(cp,r)) for r in unique([1,nside,2nside,3nside,4nside-1])]))
        for pixel in pixels
            boundary = boundariesRing(cp.resol, pixel, 4, Float64)
            theta = [atan(hypot(v[1],v[2]),v[3]) for v in eachrow(boundary)]
            phi = [mod2pi(atan(v[2],v[1])) for v in eachrow(boundary)]
            angles = Angles(cp; pix=pixel, θ=theta, φ=phi, ψ=zeros(length(theta)))
            @test maximum(angles.β) <= bound
        end
    end
end

@testset "beam-batched HDF5 pipeline" begin
    if CUDA.functional()
        # The two-GPU cases need a second device and a second Julia thread, e.g.
        # Pkg.test(julia_args=["--threads=2"]); without them only one GPU is used.
        two_gpus = length(CUDA.devices()) >= 2 && Threads.nthreads() >= 2
        pair = two_gpus ? [0, 1] : [0]
        output = mktempdir()
        cp, raw_alm, raw_blm, buffer, rejected_pixel = _pipeline_fixture()
        for T in (Float64, Float32)
            alm = prepare_alm(cp, raw_alm; precision=T)
            blm = prepare_blm(cp, raw_blm; precision=T)
            reference = _pipeline_reference(cp, buffer, alm, blm, T)
            @test all(isnan, reference[:,:,:,rejected_pixel])
            @test count(isfinite, reference) > 0
            @test count(isnan, reference) > 0
            cases = Any[(1,[0],4), (2,[0],4), (5,[0],4)]
            two_gpus && push!(cases, (2,[0,1],:auto))
            for (batch_size, devices, tile) in cases
                label = "$(T)_b$(batch_size)_g$(length(devices))"
                path = joinpath(output, label * ".h5")
                digests = Dict{Tuple{Int,Int},Vector{UInt8}}()
                batch_callback = payload -> begin
                    digests[(payload.ring,first(payload.beams))] =
                        sha256(reinterpret(UInt8, vec(payload.stokes)))
                end
                result = convolve_to_hdf5(cp, buffer, alm, blm; output_path=path,
                    beam_batch_size=batch_size, devices=devices, l_tile=tile, max_l_tile=4,
                    map_precision=T, sky_ids=["sky-a","sky-b"],
                    beam_ids=["beam-$b" for b in 1:5],
                    beam_metadata=Dict("theta_deg"=>[10.,20.,30.,40.,50.]),
                    metadata=Dict("fixture"=>"deterministic_real_alm_v1"),
                    batch_callback=batch_callback)
                @test result == path
                @test isempty(CUDA.CUFFT.idle_handles.idle_handles)
                @test isempty(CUDA.CUFFT.idle_handles.active_handles)
                actual = h5open(path,"r") do file
                    @test read(file["run_state"]) == UInt8[1]
                    @test all(==(1), read(file["rings/completed"]))
                    @test read(file["sky_ids"]) == ["sky-a","sky-b"]
                    @test read(file["beam_ids"]) == ["beam-$b" for b in 1:5]
                    @test read(file["beam_metadata/theta_deg"]) == [10.,20.,30.,40.,50.]
                    @test read(attributes(file)["disk_axes"]) == "sky,beam,stokes,pixel"
                    @test read(attributes(file)["beta_max"]) == automatic_beta_max(cp.nside_output; precision=T)
                    @test size(file["map"]) == (cp.resol.numOfPixels,3,5,2)
                    active_counts = read(file["rings/active_pixels"])
                    tiles = read(file["rings/tile_count"])
                    @test read(file["rings/scan_moment_builds"]) == Int.(active_counts .> 0)
                    @test read(file["rings/ringd_builds"]) == Int.(active_counts .> 0)
                    @test read(file["rings/sky_fft_builds"]) == tiles
                    @test read(file["rings/kernel_builds"]) == tiles
                    @test read(file["rings/beam_applications"]) == tiles .* cld(5,batch_size)
                    @test all(tiles[active_counts .> 0] .== cld(cp.lstop-cp.lstart+1,4))
                    @test read(file["rings/accumulator_bytes"]) == sizeof(Complex{T})*2*5*2 .* active_counts
                    if length(devices) == 2
                        @test sort(unique(read(file["rings/device"])[active_counts .> 0])) == [0,1]
                    end
                    permutedims(read(file["map"]), (4,3,2,1))
                end
                @test eltype(actual) === T
                for ((ring, beam_first), digest) in digests
                    beam_range = beam_first:min(beam_first+batch_size-1,cp.numofbeam)
                    saved = read_ring_hdf5(path,ring;beams=beam_range)
                    @test sha256(reinterpret(UInt8,vec(saved))) == digest
                end
                @test length(digests) == count(r -> ring_hit_count(buffer,r)>0,1:ring_count(cp))*cld(5,batch_size)
                _compare_pipeline_maps(reference, actual, T)
                @test_throws ArgumentError convolve_to_hdf5(cp,buffer,alm,blm;output_path=path)
            end
        end

        @testset "partial rings, writer errors and beta errors" begin
            T = Float64
            alm = prepare_alm(cp,raw_alm;precision=T)
            blm = prepare_blm(cp,raw_blm;precision=T)
            failure_path = joinpath(output,"writer_failure.h5")
            writer_error = try
                convolve_to_hdf5(cp,buffer,alm,blm;
                    output_path=failure_path,devices=pair,beam_batch_size=2,l_tile=4,
                    batch_callback=payload -> error("injected writer failure"))
                nothing
            catch exception
                exception
            end
            @test writer_error isa CompositeException
            @test occursin("injected writer failure", sprint(showerror, writer_error))
            h5open(failure_path,"r") do file
                @test read(file["run_state"]) == UInt8[2]
                @test file["rings/completed"][1] == 0
            end
            @test_throws ArgumentError read_ring_hdf5(failure_path,1)
            corrupt = deepcopy(buffer)
            corrupt.theta[1] = pi - corrupt.theta[1]
            beta_failure = joinpath(output,"beta_failure.h5")
            beta_error = try
                convolve_to_hdf5(cp,corrupt,alm,blm;
                    output_path=beta_failure,devices=pair,l_tile=4)
                nothing
            catch exception
                exception
            end
            @test beta_error isa CompositeException
            @test occursin("beta outside automatic interval", sprint(showerror, beta_error))
            h5open(beta_failure,"r") do file
                @test read(file["run_state"]) == UInt8[2]
                @test file["rings/completed"][1] == 0
            end
            # A successful call in the same process after both failures
            # exercises cleanup and worker joining, including empty requested rings.
            path = joinpath(output,"after_failure_subset.h5")
            convolve_to_hdf5(cp,buffer,alm,blm;output_path=path,devices=pair,
                             beam_batch_size=2,rings=[1,2,3,8],l_tile=4)
            h5open(path,"r") do file
                flags = zeros(UInt8,ring_count(cp)); flags[[1,2,3,8]] .= 1
                @test read(file["rings/completed"]) == flags
                @test read(file["rings/requested"]) == flags
                @test all(isnan, file["map"][first(ring_pixel_range(cp,4)):last(ring_pixel_range(cp,4)),:,:,:])
            end
            @test_throws ArgumentError read_ring_hdf5(path,4)
            @test all(isnan,read_ring_hdf5(path,3))
            @test size(read_ring_hdf5(path,1;skies=2:2,beams=4:5)) == (1,2,3,4)
        end
    end
end

@testset "four-detector sufficient statistics and Scanvolver U sign" begin
    intensity, q, u = 2.3, -0.7, 0.41
    base_psi = 0.37
    angles = [(; num=1, ψ=[base_psi])]
    for T in (Float32, Float64)
        tolerance = T === Float32 ? 5e-7 : 2e-14
        detector_values = zeros(Complex{T}, 1, 4, 1, 2)
        for detector in 1:4
            psi = base_psi + _DETECTOR_TEST_OFFSETS[detector]
            signal = intensity + q*cos(2psi) + u*sin(2psi)
            detector_values[1, detector, 1, 1] = signal
            detector_values[1, detector, 1, 2] = signal * cis(2base_psi)
        end
        grouped = reduce_detector_accumulators(detector_values, _DETECTOR_TEST_OFFSETS, _DETECTOR_TEST_WEIGHTS)
        expected_spin0 = intensity
        expected_spin2 = (q + im*u) / 2
        @test grouped[1, 1, 1, 1] ≈ expected_spin0 atol=tolerance
        @test grouped[1, 1, 1, 2] ≈ expected_spin2 atol=tolerance

        stokes = grouped_ring_stokes(grouped, angles, _DETECTOR_TEST_OFFSETS, _DETECTOR_TEST_WEIGHTS;
                                      rcond=1e-10, precision=T)
        @test stokes[1, 1, :, 1] ≈ T.([intensity, q, u]) atol=tolerance

        # Omitting the detector phase is not an equivalent operation.
        wrong_grouped = zeros(Complex{T}, 1, 1, 1, 2)
        wrong_grouped[1,1,1,1] = sum(detector_values[1,:,1,1]) / 4
        wrong_grouped[1,1,1,2] = sum(detector_values[1,:,1,2]) / 4
        wrong_stokes = grouped_ring_stokes(wrong_grouped, angles, _DETECTOR_TEST_OFFSETS, _DETECTOR_TEST_WEIGHTS;
                                            rcond=1e-10, precision=T)
        @test !isapprox(wrong_stokes[1,1,2,1], q; atol=1e-3)
        @test !isapprox(wrong_stokes[1,1,3,1], u; atol=1e-3)
    end
end

@testset "group validation" begin
    values = zeros(ComplexF32, 2, 8, 3, 2)
    grouped = reduce_detector_accumulators(values, _DETECTOR_TEST_OFFSETS, _DETECTOR_TEST_WEIGHTS)
    @test size(grouped) == (2, 2, 3, 2)
    @test_throws DimensionMismatch reduce_detector_accumulators(
        zeros(ComplexF32, 1, 5, 1, 2), _DETECTOR_TEST_OFFSETS, _DETECTOR_TEST_WEIGHTS,
    )
    @test_throws ArgumentError reduce_detector_accumulators(values, _DETECTOR_TEST_OFFSETS, [1, 1, 0, 1])
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
