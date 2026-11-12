# The real representation is checked against the unchanged full-complex
# implementation, including both scan weights, signed M, short polar FFTs,
# ragged l tiles, non-zero accumulators and general real Mueller mixing.
function _sr_coefficients(lmax, seed)
    raw=zeros(ComplexF64,3,(lmax+1)*(lmax+2)÷2)
    for m in 0:lmax,l in m:lmax,s in 1:3
        s>1 && l<2 && continue
        raw[s,Scanvolver._packed_alm_index(l,m,lmax)] =
            complex(sin(seed+0.31s+0.47l+0.83m),
                    m==0 ? 0.0 : cos(seed-0.27s+0.19l-0.41m))/(l+1)^2
    end
    return raw
end
_sr_relerr(a,b)=maximum(abs,a-b)/max(maximum(abs,a),maximum(abs,b),1e-30)

@testset "Spin reduction reality selection" begin
    cp=ConvolutionParams(nside_output=4,lmax=6,lstop=6)
    prepared=prepare_alm(cp,_sr_coefficients(6,0.2))
    @test Scanvolver._resolve_spin_reduction(prepared,:auto)
    @test Scanvolver._resolve_spin_reduction(prepared,:on)
    @test !Scanvolver._resolve_spin_reduction(prepared,:off)
    @test_throws ArgumentError Scanvolver._resolve_spin_reduction(prepared,:invalid)
    # Input-basis :spin is also eligible; no B=0 or E-only assumption.
    raw=_sr_coefficients(6,0.2)
    rawspin=copy(raw)
    for j in axes(raw,2)
        rawspin[:,j].=Scanvolver._positive_spin_coefficients(raw,j,:teb,Float64)
    end
    @test Scanvolver._resolve_spin_reduction(prepare_alm(cp,rawspin;input_basis=:spin),:auto)
    # m=0 is not repaired or projected silently.
    invalid=prepare_alm(cp,_sr_coefficients(6,0.2))
    invalid.slices[1][1][1,1]+=im
    @test !Scanvolver._resolve_spin_reduction(invalid,:auto)
    @test_throws ArgumentError Scanvolver._resolve_spin_reduction(invalid,:on)
    # A manually modified nonzero-m polarization also disables reduction.
    invalid=prepare_alm(cp,_sr_coefficients(6,0.2))
    invalid.slices[4][3][1,2]+=0.1im
    @test !Scanvolver._resolve_spin_reduction(invalid,:auto)
    @test_throws ArgumentError Scanvolver._resolve_spin_reduction(invalid,:on)

    full=ring_memory_bytes(cp,16,8,3)
    reduced=ring_memory_bytes(cp,16,8,3;spin_reduction=true)
    @test reduced.sky_ring*2==full.sky_ring
    @test reduced.active_sky_ring*2==full.active_sky_ring
    @test reduced.effective_beam==full.effective_beam
    @test reduced.sky_fft>0
    @test reduced.contraction_workspace>0
    @test streamed_tile_bytes(cp,16,8,3;spin_reduction=true) >=
        reduced.sky_fft+reduced.sky_ring
end

function _sr_compare_tiles(T,hwp,ring)
    cp=ConvolutionParams(nside_output=4,lmax=6,lstop=6,bnmax=3,κ=4,
        numofsky=2,numofbeam=2,HWP=hwp,HWP_rpm=hwp ? 0.0 : nothing)
    a=prepare_alm(cp,[_sr_coefficients(6,0.2),_sr_coefficients(6,1.7)];precision=T)
    rawbeams=[_sr_coefficients(6,0.8),_sr_coefficients(6,2.3)]
    M=[sin(0.73i+0.41j)+0.4cos(0.19i-0.67j) for i in 1:4,j in 1:4]
    b=hwp ? prepare_hwp_blm(cp,rawbeams;precision=T,mueller=M,lowell_hwp=:full_sum) :
        prepare_blm(cp,rawbeams;precision=T)
    cg=Scanvolver._geometry_params(cp,b)
    g=ring_geometry(cg,ring)
    positions=ring==1 ? collect(1:g.pixel_count) : collect(1:2:g.pixel_count)
    angles=[Angles(cp;pix=g.first_pixel+p-1,
        φ=[g.phi_pixels[p]+0.002sin(0.7j) for j in 1:13],
        θ=[g.theta+0.002cos(0.9j) for j in 1:13],
        ψ=[0.43j+0.17p for j in 1:13],
        hwp_angle=hwp ? [mod(0.71j+0.23p,2pi) for j in 1:13] : nothing) for p in positions]
    plan=ScanMomentPlan(kappa=cg.κ,nmax=calc_nmax(cg,6),beta_order=12,beta_max=0.02,precision=T)
    table=upload_wigner_table_gpu(build_wigner_chebyshev_table_cpu(cg,plan))
    fullalm=upload_alm_gpu(a)
    reducedalm=upload_alm_gpu(a;spin_reduction=true)
    @test size(fullalm.values,2)==3
    @test size(reducedalm.values,2)==2
    @test length(reducedalm.values)*3==length(fullalm.values)*2
    context=ringd_context(cg;precision=T)
    ringd=ringd_blocks_recursion_gpu!(context,cg,g.theta)
    active=ActiveRingPixels(positions,g.pixel_count)
    components=hwp ? b.components : [b]
    modes=hwp ? b.modes : [:constant]
    moments=[scan_moments_gpu(plan,angles;hwp_mode=mode,demodulate_hwp=hwp) for mode in modes]
    initial=fill(Complex{T}(0.031,-0.017),2,2,length(positions),2)
    fullacc=CuArray(initial); reducedacc=CuArray(initial)
    tolerance=T===Float64 ? 3e-11 : 4e-4
    try
        for first in 0:3:6
            last=min(first+2,6)
            full=build_sky_ring_gpu(cg,ringd,fullalm,g.phi_pixels,first,last)
            reduced=build_sky_ring_gpu(cg,ringd,reducedalm,g.phi_pixels,first,last)
            afull=select_active_sky_ring_gpu(full,active)
            areduced=select_active_sky_ring_gpu(reduced,active)
            @test eltype(reduced)===T
            @test sizeof(T)*length(reduced)*2==sizeof(Complex{T})*length(full)
            @test (areduced===reduced)==active.complete
            estimate=ring_memory_bytes(cg,g.pixel_count,length(active),last-first+1;
                precision=T,complete_ring=active.complete,spin_reduction=true)
            @test estimate.sky_ring==sizeof(T)*length(reduced)
            @test estimate.active_sky_ring==(active.complete ? 0 : sizeof(T)*length(areduced))
            for (component,moment) in zip(components,moments)
                kernel=build_effective_kernel_gpu(table,moment,first,last)
                efull=apply_beams_gemm_gpu(table,kernel,component,first,last)
                ered=apply_beams_gemm_gpu(table,kernel,component,first,last;spin_reduction=true)
                @test eltype(ered)===T
                @test sizeof(T)*length(ered)==sizeof(Complex{T})*length(efull)
                accumulate_convolution_tile_gemm_gpu!(fullacc,afull,efull)
                accumulate_convolution_tile_gemm_gpu!(reducedacc,areduced,ered)
                @test _sr_relerr(Array(fullacc),Array(reducedacc))<tolerance
                foreach(CUDA.unsafe_free!,(kernel,efull,ered))
            end
            free_active_sky_ring!(afull,full)
            free_active_sky_ring!(areduced,reduced)
            foreach(CUDA.unsafe_free!,(full,reduced))
        end
        @test _sr_relerr(Array(fullacc),Array(reducedacc))<tolerance
        # The last tile has width 1, and signs/scan weights must still match.
        @test size(reducedalm.values,2)==2
    finally
        foreach(CUDA.unsafe_free!,moments)
        foreach(CUDA.unsafe_free!,(fullacc,reducedacc,fullalm.values,reducedalm.values,active.positions))
        CUDA.unsafe_free!(table.coefficients); CUDA.unsafe_free!(table.pair_lookup)
        free_ringd_context!(context)
    end
end

@testset "Spin reduction GPU tiles and general Mueller" begin
    if CUDA.functional()
        for T in (Float64,Float32), hwp in (false,true), ring in (1,7)
            _sr_compare_tiles(T,hwp,ring)
        end
    else
        @test_skip CUDA.functional()
    end
end

@testset "Spin reduction HDF5 selection and detector groups" begin
    if CUDA.functional()
        cp=ConvolutionParams(nside_output=4,lmax=6,lstop=6,bnmax=3,κ=8,numofbeam=4)
        g=ring_geometry(cp,7)
        phi=Float64[];theta=Float64[];psi=Float64[]
        for p in (1,3),j in 1:24
            push!(phi,g.phi_pixels[p]+0.002sin(0.7j))
            push!(theta,g.theta+0.002cos(0.9j))
            push!(psi,0.43j+0.17p)
        end
        buffer=build_ring_scan_buffer(cp,phi,theta,psi)
        sky=prepare_alm(cp,_sr_coefficients(6,0.2))
        beams=prepare_blm(cp,[_sr_coefficients(6,x) for x in (0.3,0.7,1.2,2.1)])
        mktempdir() do dir
            for grouped in (false,true)
                paths=[joinpath(dir,"$(grouped)_$(mode).h5") for mode in (:off,:on)]
                driver=grouped ? convolve_detector_groups_to_hdf5 : convolve_to_hdf5
                for (path,mode) in zip(paths,(:off,:on))
                    driver(cp,buffer,sky,beams;output_path=path,rings=[7],
                        spin_reduction=mode,l_tile=3,beta_order=12,rcond=1e-8)
                    h5open(path,"r") do f
                        @test read(HDF5.attributes(f)["spin_reduction"])==(mode===:on)
                        @test read(f["run_state"])==UInt8[1]
                    end
                end
                a,b=(h5open(f->read(f["map"]),p,"r") for p in paths)
                @test isnan.(a)==isnan.(b)
                mask=isfinite.(a)
                @test any(mask)
                @test _sr_relerr(a[mask],b[mask])<3e-11
            end
            invalid=prepare_alm(cp,_sr_coefficients(6,0.2))
            invalid.slices[1][1][1,1]+=im
            path=joinpath(dir,"fallback.h5")
            convolve_to_hdf5(cp,buffer,invalid,beams;output_path=path,rings=[7],spin_reduction=:auto)
            h5open(path,"r") do f
                @test !read(HDF5.attributes(f)["spin_reduction"])
            end
            @test_throws ArgumentError convolve_to_hdf5(cp,buffer,invalid,beams;
                output_path=joinpath(dir,"invalid.h5"),rings=[7],spin_reduction=:on)
            @test !ispath(joinpath(dir,"invalid.h5"))
        end
    else
        @test_skip CUDA.functional()
    end
end
