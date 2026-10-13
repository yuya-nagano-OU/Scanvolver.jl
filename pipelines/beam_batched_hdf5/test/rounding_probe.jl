using CUDA, EBCoDeCMB, Healpix, NPZ, TOML, Printf
include(joinpath(@__DIR__, "..", "beam_batched_pipeline.jl"))
using .BeamBatchedHDF5

# Reuse only the two fixture functions from the representative regression,
# without running its testset or creating its full-sky maps.
for expression in Meta.parseall(read(joinpath(@__DIR__, "representative.jl"),String)).args
    if expression isa Expr && expression.head === :function
        name = expression.args[1].args[1]
        name in (:truncate_packed,:corner_scan) && Core.eval(Main,expression)
    end
end
const ROOT = get(() -> error("set EBCODE_INPUT_ROOT to the directory that holds inputs/ and EBCoDe_gpu_recursion/"), ENV, "EBCODE_INPUT_ROOT")
const RINGS = [1,2,30,256,512,768,1022,1023]
const SKY_PATHS = [joinpath(ROOT,"inputs","CMB","alm=CMB=r0=nside512=seed$(s).npy") for s in 0:1]
const BEAM_DIR = joinpath(ROOT,"EBCoDe_gpu_recursion","test","performance","inputs",
                         "rotated_gaussian_fwhm5_theta10_90_phi0_lmax200_nside512")
const BEAM_PATHS = sort(filter(p -> endswith(p,".npy"),readdir(BEAM_DIR;join=true)))[1:5]

maxdiff(a,b) = maximum(abs,ComplexF64.(a) .- ComplexF64.(b))

function probe()
    CUDA.device!(0)
    cp = ConvolutionParams(nside_output=256,lmax=200,lstop=200,bnmax=120,κ=20,numofsky=2,numofbeam=5)
    alm = prepare_alm(cp,truncate_packed.(SKY_PATHS,200);precision=Float32)
    blm = prepare_blm(cp,truncate_packed.(BEAM_PATHS,200);precision=Float32)
    buffer = corner_scan(cp)
    plan = automatic_scan_plan(cp;precision=Float32,beta_order=8)
    table = build_wigner_chebyshev_table_cpu(cp,plan)
    batches = beam_batches(blm,2)
    for batch in batches, l in eachindex(blm.slices), s in 1:3
        @assert isequal(batch.blm.slices[l][s],blm.slices[l][s][:,batch.indices])
    end
    context = BeamBatchedHDF5.prepare_context(cp,alm,table,0,Float32)
    rows = Any[]
    try
        for ring in RINGS
            geo = ring_geometry(cp,ring)
            active = moments = nothing
            owned = CuArray[]
            track(x) = (push!(owned,x); x)
            effective_error = control_error = actual_error = 0.0
            first_effective_tile = first_control_tile = -1
            first_control_error = 0.0
            try
                active, angles = make_ring_angles(cp,geo,buffer)
                ringd = ringd_blocks_recursion_gpu!(context.ringd,cp,geo.theta)
                moments = scan_moments_gpu(plan,angles;method=:gemm)
                full = track(CUDA.zeros(ComplexF32,2,5,length(active),2))
                controls = [track(CUDA.zeros(ComplexF32,2,length(b.indices),length(active),2)) for b in batches]
                actual = [track(CUDA.zeros(ComplexF32,2,length(b.indices),length(active),2)) for b in batches]
                for lf in 0:4:200
                    ll = min(lf+3,200)
                    kernel = sky = u = full_e = nothing
                    try
                        kernel = build_effective_kernel_gpu(context.table,moments,lf,ll)
                        sky = build_sky_ring_gpu(cp,ringd,context.alm,geo.phi_pixels,lf,ll)
                        u = select_active_sky_ring_gpu(sky,active)
                        full_e = apply_beams_gemm_gpu(context.table,kernel,blm,lf,ll)
                        ef = Array(full_e)
                        accumulate_convolution_tile_gemm_gpu!(full,u,full_e)
                        cf = Array(full)
                        for (b,control,got) in zip(batches,controls,actual)
                            partial_e = fixed_e = nothing
                            try
                                partial_e = apply_beams_gemm_gpu(context.table,kernel,b.blm,lf,ll)
                                expected_e = ef[:,b.indices,:,:]
                                de = maxdiff(Array(partial_e),expected_e)
                                effective_error = max(effective_error,de)
                                de > 0 && first_effective_tile < 0 && (first_effective_tile=lf)
                                # Feed a bit-for-bit subset of the full-beam E,
                                # so this control excludes beam-application errors.
                                fixed_e = CuArray(expected_e)
                                @assert reinterpret(UInt8,vec(Array(fixed_e))) == reinterpret(UInt8,vec(expected_e))
                                accumulate_convolution_tile_gemm_gpu!(control,u,fixed_e)
                                accumulate_convolution_tile_gemm_gpu!(got,u,partial_e)
                                dc = maxdiff(Array(control),cf[:,b.indices,:,:])
                                control_error = max(control_error,dc)
                                actual_error = max(actual_error,maxdiff(Array(got),cf[:,b.indices,:,:]))
                                if dc > 0 && first_control_tile < 0
                                    first_control_tile = lf
                                    first_control_error = dc
                                end
                            finally
                                partial_e === nothing || CUDA.unsafe_free!(partial_e)
                                fixed_e === nothing || CUDA.unsafe_free!(fixed_e)
                            end
                        end
                    finally
                        CUDA.synchronize()
                        full_e === nothing || CUDA.unsafe_free!(full_e)
                        u === nothing || free_active_sky_ring!(u,sky)
                        sky === nothing || CUDA.unsafe_free!(sky)
                        kernel === nothing || CUDA.unsafe_free!(kernel)
                    end
                end
                row = Dict("ring"=>ring,"max_effective_beam_difference"=>effective_error,
                    "max_accumulator_difference_with_identical_E"=>control_error,
                    "max_accumulator_difference_with_batched_E"=>actual_error,
                    "first_effective_difference_lfirst"=>first_effective_tile,
                    "first_control_difference_lfirst"=>first_control_tile,
                    "first_control_difference"=>first_control_error)
                push!(rows,row)
                @printf("RING %4d E=%.9e C_same_E=%.9e C_batched_E=%.9e first_E_l=%d first_C_l=%d\n",
                    ring,effective_error,control_error,actual_error,first_effective_tile,first_control_tile)
                flush(stdout)
            finally
                CUDA.synchronize()
                foreach(CUDA.unsafe_free!,owned)
                moments === nothing || CUDA.unsafe_free!(moments)
                active === nothing || CUDA.unsafe_free!(active.positions)
            end
        end
    finally
        BeamBatchedHDF5.free_context(context)
    end
    output = joinpath(@__DIR__,"..","test-output","rounding_probe_"*string(time_ns())*".toml")
    open(io -> TOML.print(io,Dict("precision"=>"Float32","beam_batch_size"=>2,
        "nside"=>256,"lmax"=>200,"bnmax"=>120,"kappa"=>20,"l_tile"=>4,
        "rows"=>rows,"status"=>"completed")),output,"w")
    println("PROBE_OUTPUT="*abspath(output))
end
probe()
