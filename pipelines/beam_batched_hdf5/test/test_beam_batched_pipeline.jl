using Test, CUDA, EBCoDeCMB, Healpix, HDF5, LinearAlgebra, SHA, TOML, Printf
using HDF5: attributes
using NPZ
include(joinpath(@__DIR__, "..", "beam_batched_pipeline.jl"))
using .BeamBatchedHDF5

# Declared before evaluating any new-pipeline result. These are pointwise
# tolerances on normalized test signals, not fitted to measured differences.
const TOLERANCES = Dict(Float32 => (atol=2e-6, rtol=2e-5),
                        Float64 => (atol=1e-12, rtol=1e-11))
const OUTPUT = abspath(get(ENV, "EBCODE_PIPELINE_TEST_OUTPUT",
                   joinpath(@__DIR__, "..", "test-output", "run_" * string(time_ns()))))
mkpath(OUTPUT)
const REPORT = Dict{String,Any}("tolerances" => Dict(string(k) => Dict("atol"=>v.atol, "rtol"=>v.rtol)
                                                 for (k,v) in TOLERANCES),
                                 "cases" => Any[])

function fixture(; nside=8, lmax=12, bnmax=4, kappa=3)
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

function baseline(cp, buffer, alm, blm, T; tile=4)
    CUDA.device!(0)
    plan = automatic_scan_plan(cp; precision=T)
    table = upload_wigner_table_gpu(build_wigner_chebyshev_table_cpu(cp, plan))
    gpu_alm = nothing
    try
        gpu_alm = upload_alm_gpu(alm)
        return convolve_sphere_gpu(cp, buffer, gpu_alm, blm, table, plan;
                                   l_tile=tile, map_precision=T, rcond=1e-2,
                                   moment_method=:gemm)
    finally
        gpu_alm === nothing || CUDA.unsafe_free!(gpu_alm.values)
        CUDA.unsafe_free!(table.coefficients)
        CUDA.unsafe_free!(table.pair_lookup)
        CUDA.reclaim()
    end
end

function compare_maps(reference, actual, T, label)
    tol = TOLERANCES[T]
    @test size(reference) == size(actual)
    @test isnan.(reference) == isnan.(actual)
    @test isfinite.(reference) == isfinite.(actual)
    rows = Any[]
    for (stokes, name) in enumerate(("I","Q","U"))
        ref = Float64.(vec(reference[:,:,stokes,:]))
        got = Float64.(vec(actual[:,:,stokes,:]))
        valid = isfinite.(ref)
        @test count(valid) > 0
        delta = abs.(got[valid] .- ref[valid])
        @test all(delta .<= tol.atol .+ tol.rtol .* abs.(ref[valid]))
        maximum_absolute = maximum(delta)
        relative_l2 = norm(delta) / norm(ref[valid])
        maximum_scaled = maximum(delta ./ (tol.atol .+ tol.rtol .* abs.(ref[valid])))
        push!(rows, Dict("stokes"=>name, "finite_values"=>count(valid),
                         "max_absolute_error"=>maximum_absolute,
                         "relative_l2_error"=>relative_l2,
                         "max_tolerance_ratio"=>maximum_scaled))
        @printf("COMPARE %-24s %s max_abs=%.6e relative_l2=%.6e tolerance_ratio=%.6e\n",
                label, name, maximum_absolute, relative_l2, maximum_scaled)
    end
    return rows
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
    @test CUDA.functional()
    @test length(CUDA.devices()) >= 2
    @test Threads.nthreads() >= 2
    cp, raw_alm, raw_blm, buffer, rejected_pixel = fixture()
    for T in (Float64, Float32)
        alm = prepare_alm(cp, raw_alm; precision=T)
        blm = prepare_blm(cp, raw_blm; precision=T)
        reference = baseline(cp, buffer, alm, blm, T)
        npzwrite(joinpath(OUTPUT, "baseline_$(T).npy"), reference)
        @test all(isnan, reference[:,:,:,rejected_pixel])
        @test count(isfinite, reference) > 0
        @test count(isnan, reference) > 0
        for (batch_size, devices, tile) in ((1,[0],4), (2,[0],4), (5,[0],4), (2,[0,1],:auto))
            label = "$(T)_b$(batch_size)_g$(length(devices))"
            path = joinpath(OUTPUT, label * ".h5")
            digests = Dict{Tuple{Int,Int},Vector{UInt8}}()
            batch_callback = payload -> begin
                digests[(payload.ring,first(payload.beams))] =
                    sha256(reinterpret(UInt8, vec(payload.stokes)))
            end
            started = time()
            result = convolve_to_hdf5(cp, buffer, alm, blm; output_path=path,
                beam_batch_size=batch_size, devices=devices, l_tile=tile, max_l_tile=4,
                map_precision=T, sky_ids=["sky-a","sky-b"],
                beam_ids=["beam-$b" for b in 1:5],
                beam_metadata=Dict("theta_deg"=>[10.,20.,30.,40.,50.]),
                metadata=Dict("fixture"=>"deterministic_real_alm_v1"),
                batch_callback=batch_callback)
            elapsed = time() - started
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
            errors = compare_maps(reference,actual,T,label)
            push!(REPORT["cases"],Dict("case"=>label,"seconds"=>elapsed,"path"=>path,
                                      "beam_batch_size"=>batch_size,"devices"=>devices,"errors"=>errors))
            @test_throws ArgumentError convolve_to_hdf5(cp,buffer,alm,blm;output_path=path)
        end
    end

    @testset "partial rings, writer errors and beta errors" begin
        T = Float64
        alm = prepare_alm(cp,raw_alm;precision=T)
        blm = prepare_blm(cp,raw_blm;precision=T)
        failure_path = joinpath(OUTPUT,"writer_failure.h5")
        writer_error = try
            convolve_to_hdf5(cp,buffer,alm,blm;
                output_path=failure_path,devices=[0,1],beam_batch_size=2,l_tile=4,
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
        beta_failure = joinpath(OUTPUT,"beta_failure.h5")
        beta_error = try
            convolve_to_hdf5(cp,corrupt,alm,blm;
                output_path=beta_failure,devices=[0,1],l_tile=4)
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
        # A successful two-GPU call in the same process after both failures
        # exercises cleanup and worker joining, including empty requested rings.
        path = joinpath(OUTPUT,"after_failure_subset.h5")
        convolve_to_hdf5(cp,buffer,alm,blm;output_path=path,devices=[0,1],
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

REPORT["julia_version"] = string(VERSION)
REPORT["cuda_version"] = string(pkgversion(CUDA))
REPORT["hdf5_version"] = string(pkgversion(HDF5))
REPORT["ebcodecmb_source"] = pathof(EBCoDeCMB)
REPORT["status"] = "passed"
open(joinpath(OUTPUT,"metrics.toml"),"w") do io
    TOML.print(io,REPORT)
end
println("TEST_OUTPUT=" * OUTPUT)
