using Test, CUDA, EBCoDeCMB, Healpix, HDF5, NPZ, LinearAlgebra, SHA, TOML, Printf
using HDF5: attributes
include(joinpath(@__DIR__, "..", "beam_batched_pipeline.jl"))
using .BeamBatchedHDF5

# Fix the normalized tolerances before running either implementation. Actual
# absolute tolerances are scaled by the reference Stokes amplitude (native units).
const ATOL_FRACTION = 2e-6
const RTOL = 2e-5
const ROOT = get(() -> error("set EBCODE_INPUT_ROOT to the directory that holds inputs/ and EBCoDe_gpu_recursion/"), ENV, "EBCODE_INPUT_ROOT")
const OUTPUT = abspath(get(ENV, "EBCODE_PIPELINE_TEST_OUTPUT",
    joinpath(@__DIR__, "..", "test-output", "representative_" * string(time_ns()))))
const SKY_PATHS = [joinpath(ROOT,"inputs","CMB","alm=CMB=r0=nside512=seed$(s).npy") for s in 0:1]
const BEAM_DIR = joinpath(ROOT,"EBCoDe_gpu_recursion","test","performance","inputs",
                         "rotated_gaussian_fwhm5_theta10_90_phi0_lmax200_nside512")
const BEAM_PATHS = sort(filter(p -> endswith(p,".npy"),readdir(BEAM_DIR;join=true)))[1:5]
const RINGS = [1,2,30,256,512,768,1022,1023]
mkpath(OUTPUT)

# The current prepare APIs require exactly cp.lmax. Repack the existing packed
# HEALPix arrays in this fixture; do not change the package's input API.
function truncate_packed(path, target_lmax)
    source = npzread(path)
    @assert ndims(source) == 2 && size(source,1) == 3
    source_lmax = (isqrt(8size(source,2)+1)-3) ÷ 2
    @assert (source_lmax+1)*(source_lmax+2) ÷ 2 == size(source,2)
    @assert source_lmax >= target_lmax
    out = similar(source,3,(target_lmax+1)*(target_lmax+2) ÷ 2)
    for m in 0:target_lmax
        source_first = m*(2source_lmax+1-m) ÷ 2 + m + 1
        dest_first = m*(2target_lmax+1-m) ÷ 2 + m + 1
        count = target_lmax-m+1
        out[:,dest_first:dest_first+count-1] = source[:,source_first:source_first+count-1]
    end
    return out
end

function corner_scan(cp)
    phi, theta, psi = Float64[], Float64[], Float64[]
    for ring in RINGS
        geo = ring_geometry(cp,ring)
        positions = unique(round.(Int,range(1,geo.pixel_count;length=min(16,geo.pixel_count))))
        for position in positions
            pixel = geo.first_pixel + position - 1
            corners = boundariesRing(cp.resol,pixel,1,Float64)
            center = [sin(geo.theta)*cos(geo.phi_pixels[position]),
                      sin(geo.theta)*sin(geo.phi_pixels[position]),cos(geo.theta)]
            for hit in 0:7
                corner = corners[mod(hit,4)+1,:]
                fraction = hit == 0 ? 0.0 : 0.995
                target_theta = target_phi = 0.0
                for attempt in 1:12
                    v = (1-fraction)*center + fraction*corner
                    target_theta = atan(hypot(v[1],v[2]),v[3])
                    target_phi = mod2pi(atan(v[2],v[1]))
                    ang2pixRing(cp.resol,target_theta,target_phi) == pixel && break
                    fraction *= 0.9
                end
                @assert ang2pixRing(cp.resol,target_theta,target_phi) == pixel
                push!(phi,target_phi); push!(theta,target_theta); push!(psi,2pi*hit/8)
            end
        end
    end
    return build_ring_scan_buffer(cp,phi,theta,psi)
end

const REPORT = Dict{String,Any}("status"=>"pending", "atol_fraction"=>ATOL_FRACTION,
    "rtol"=>RTOL,"nside"=>256,"lmax"=>200,"bnmax"=>120,"kappa"=>20,
    "beta_order"=>8,"l_tile"=>4,"skies"=>2,"beams"=>5,"rings"=>RINGS,
    "scan"=>"synthetic near-corner pointings, eight orientations per pixel",
    "inputs"=>[Dict("path"=>p,"sha256"=>bytes2hex(open(sha256,p))) for p in vcat(SKY_PATHS,BEAM_PATHS)])

@testset "nside256 existing sky and rotated beams" begin
    cp = ConvolutionParams(nside_output=256,lmax=200,lstop=200,bnmax=120,κ=20,numofsky=2,numofbeam=5)
    alm = prepare_alm(cp,truncate_packed.(SKY_PATHS,200);precision=Float32)
    blm = prepare_blm(cp,truncate_packed.(BEAM_PATHS,200);precision=Float32)
    buffer = corner_scan(cp)
    plan = automatic_scan_plan(cp;precision=Float32,beta_order=8)
    println("Preparing reference Wigner table; beta_max=",plan.beta_max); flush(stdout)
    table = build_wigner_chebyshev_table_cpu(cp,plan)
    started = time()
    reference = convolve_sphere_multigpu(cp,buffer,alm,blm,table,plan;
        devices=[0,1],l_tile=4,map_precision=Float32,rcond=1e-2,moment_method=:gemm)
    REPORT["reference_seconds"] = time()-started
    println("Reference complete in ",REPORT["reference_seconds"]," seconds"); flush(stdout)
    npzwrite(joinpath(OUTPUT,"baseline_Float32.npy"),reference)
    path = joinpath(OUTPUT,"Float32_b2_g2.h5")
    digests = Dict{Tuple{Int,Int},Vector{UInt8}}()
    started = time()
    convolve_to_hdf5(cp,buffer,alm,blm;output_path=path,devices=[0,1],rings=RINGS,
        beam_batch_size=2,l_tile=4,beta_order=8,map_precision=Float32,
        sky_ids=basename.(SKY_PATHS),beam_ids=basename.(BEAM_PATHS),
        beam_metadata=Dict("theta_deg"=>collect(10.:10.:50.),"phi_deg"=>zeros(5),
                           "fwhm_deg"=>fill(5.,5),"generation_nside"=>fill(512,5)),
        metadata=Dict("scan"=>REPORT["scan"],"sky_paths"=>SKY_PATHS,"beam_paths"=>BEAM_PATHS),
        batch_callback=p -> (digests[(p.ring,first(p.beams))]=sha256(reinterpret(UInt8,vec(p.stokes)))),
        progress_callback=r -> (println("Saved ring ",r.ring," on GPU ",r.device);flush(stdout)))
    REPORT["pipeline_seconds"] = time()-started
    actual = h5open(path,"r") do file
        @test read(file["run_state"]) == UInt8[1]
        expected = zeros(UInt8,ring_count(cp)); expected[RINGS] .= 1
        @test read(file["rings/requested"]) == expected
        @test read(file["rings/completed"]) == expected
        @test read(file["sky_ids"]) == basename.(SKY_PATHS)
        @test read(file["beam_ids"]) == basename.(BEAM_PATHS)
        @test read(file["rings/scan_moment_builds"])[RINGS] == ones(Int,8)
        @test read(file["rings/ringd_builds"])[RINGS] == ones(Int,8)
        @test read(file["rings/tile_count"])[RINGS] == fill(51,8)
        @test read(file["rings/kernel_builds"])[RINGS] == fill(51,8)
        @test read(file["rings/sky_fft_builds"])[RINGS] == fill(51,8)
        @test read(file["rings/beam_applications"])[RINGS] == fill(153,8)
        @test sort(unique(read(file["rings/device"])[RINGS])) == [0,1]
        observed = maximum(read(file["rings/observed_beta_max"])[RINGS])
        @test 0.004 < observed <= plan.beta_max
        REPORT["beta_max"] = Float64(plan.beta_max)
        REPORT["observed_beta_max"] = observed
        permutedims(read(file["map"]),(4,3,2,1))
    end
    @test size(actual) == size(reference)
    @test isnan.(actual) == isnan.(reference)
    @test isfinite.(actual) == isfinite.(reference)
    @test length(digests) == 8*3
    for ((ring,beam_first),digest) in digests
        saved = read_ring_hdf5(path,ring;beams=beam_first:min(beam_first+1,5))
        @test sha256(reinterpret(UInt8,vec(saved))) == digest
    end
    errors = Any[]
    for (s,name) in enumerate(("I","Q","U"))
        ref, got = Float64.(vec(reference[:,:,s,:])), Float64.(vec(actual[:,:,s,:]))
        valid = isfinite.(ref)
        @test count(valid) > 0
        scale = maximum(abs,ref[valid])
        @test scale > 0
        atol = ATOL_FRACTION*scale
        delta = abs.(got[valid]-ref[valid])
        ratio = maximum(delta ./ (atol .+ RTOL .* abs.(ref[valid])))
        @test ratio <= 1
        row = Dict("stokes"=>name,"max_absolute_error"=>maximum(delta),
            "relative_l2_error"=>norm(delta)/norm(ref[valid]),"max_tolerance_ratio"=>ratio,
            "absolute_tolerance"=>atol,"reference_amplitude"=>scale,"finite_values"=>count(valid))
        push!(errors,row)
        @printf("COMPARE representative %s max_abs=%.6e relative_l2=%.6e tolerance_ratio=%.6e\n",
            name,row["max_absolute_error"],row["relative_l2_error"],ratio)
    end
    REPORT["errors"] = errors
end
REPORT["status"] = "passed"
REPORT["julia_version"] = string(VERSION)
REPORT["ebcodecmb_source"] = pathof(EBCoDeCMB)
open(io -> TOML.print(io,REPORT),joinpath(OUTPUT,"metrics.toml"),"w")
println("TEST_OUTPUT="*OUTPUT)
