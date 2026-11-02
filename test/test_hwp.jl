using Scanvolver
using CUDA
using Test
using LinearAlgebra: I
using HDF5: h5open, attributes

const _HWP_IDEAL_TEST = [1.0 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]

function hwp_test_coefficients(lmax, seed)
    result = zeros(ComplexF64, 3, (lmax + 1) * (lmax + 2) ÷ 2)
    for m in 0:lmax, l in m:lmax, s in 1:3
        s > 1 && l < 2 && continue
        j = Scanvolver._packed_alm_index(l, m, lmax)
        result[s, j] = complex(sin(seed + 0.31s + 0.47l + 0.83m),
            m == 0 ? 0.0 : cos(seed - 0.27s + 0.19l - 0.41m)) / (l + 1)^2
    end
    return result
end

# Direct finite Wigner sum, independent of the Chebyshev table, local Euler
# split, FFT and GPU recurrence. This oracle is deliberately restricted to l<=8.
function hwp_test_wignerd(l, m, n, theta)
    prefactor = sqrt(Float64(factorial(l+m) * factorial(l-m)) *
                     Float64(factorial(l+n) * factorial(l-n)))
    return sum((-1.0)^k * prefactor /
        (Float64(factorial(l+m-k)) * factorial(k) * factorial(n-m+k) * factorial(l-n-k)) *
        cos(theta/2)^(2l+m-n-2k) * sin(theta/2)^(n-m+2k)
        for k in max(0,m-n):min(l+m,l-n))
end

function hwp_test_direct_tod(cp, rawsky, prepared, phi, theta, psi, eta)
    sky = prepare_alm(cp, rawsky)
    tod = zeros(ComplexF64, length(phi), cp.numofbeam)
    for i in eachindex(phi), l in cp.lstart:cp.lstop
        a = alm_slice(sky, l)
        nmax = min(prepared.effective_bnmax, l)
        for (mode, component) in zip(prepared.modes, prepared.components)
            b = blm_slice(component, l)
            modulation = Scanvolver._hwp_modulation(mode, eta[i])
            for m in -l:l, n in -nmax:nmax
                rotation = hwp_test_wignerd(l,m,n,theta[i]) * cis(m*phi[i]+n*psi[i]) * modulation
                for s in 1:3, beam in 1:cp.numofbeam
                    tod[i,beam] += (s == 1 ? 1.0 : 0.5) * a[s][1,m+l+1] *
                        b[s][n+nmax+1,beam] * rotation
                end
            end
        end
    end
    return tod
end

function hwp_test_scan(cp)
    phi, theta, psi, eta = Float64[], Float64[], Float64[], Float64[]
    pixels = Int[]
    rings = [4, 7, 11]
    for ring in rings
        g = ring_geometry(cp, ring)
        for position in (1, 3), hit in 1:16
            push!(phi, g.phi_pixels[position] + 2e-4sin(0.7hit))
            push!(theta, g.theta + 1e-4cos(0.9hit))
            push!(psi, 0.43hit + 0.17position)
            push!(eta, mod(0.71hit + 0.23position + 0.13ring, 2pi))
            push!(pixels, g.first_pixel + position - 1)
        end
    end
    return (; phi, theta, psi, eta, pixels, rings)
end

@testset "HWP phase transport and Fourier beams" begin
    cp = ConvolutionParams(nside_output=4,lmax=6,lstop=6,bnmax=3,κ=12,
                           HWP=true,HWP_rpm=46.)
    scan = hwp_test_scan(cp)
    buffer = build_ring_scan_buffer(cp,scan.phi,scan.theta,scan.psi; hwp_angle=scan.eta)
    @test length(buffer.hwp_angle) == length(scan.eta)
    @test ring_scan_buffer_bytes(buffer) == sum(sizeof, (buffer.phi,buffer.theta,
        buffer.psi,buffer.hwp_angle,buffer.pixel_offsets))
    # Pixel bucketing must move the HWP phase with its original sample.
    for j in eachindex(buffer.phi)
        i = findfirst(k -> scan.phi[k] == buffer.phi[j] && scan.theta[k] == buffer.theta[j],eachindex(scan.phi))
        @test buffer.hwp_angle[j] == scan.eta[i]
    end
    chunks = [(phi=scan.phi[r],theta=scan.theta[r],psi=scan.psi[r],time_array=Float64.(r))
              for r in (1:31,32:length(scan.phi))]
    timed = build_ring_scan_buffer(cp,chunks; hwp_phase_rad=0.37)
    for j in eachindex(timed.phi)
        i = findfirst(k -> scan.phi[k] == timed.phi[j] && scan.theta[k] == timed.theta[j],eachindex(scan.phi))
        @test timed.hwp_angle[j] ≈ mod(2pi*46/60*i+0.37,2pi)
    end
    @test_throws ArgumentError build_ring_scan_buffer(cp,scan.phi,scan.theta,scan.psi)
    @test_throws ArgumentError build_ring_scan_buffer(cp,scan.phi,scan.theta,scan.psi;hwp_angle=[NaN])
    @test_throws ArgumentError build_ring_scan_buffer(ConvolutionParams(cp;HWP=false),
        scan.phi,scan.theta,scan.psi;hwp_angle=scan.eta)
    @test_throws ArgumentError prepare_hwp_blm(cp,zeros(3,28);mueller=zeros(3,3))
    keyed=build_ring_scan_buffer(cp,[(hwp_angle=scan.eta,psi=scan.psi,theta=scan.theta,phi=scan.phi)])
    @test keyed.hwp_angle == buffer.hwp_angle
    @test keyed.phi == buffer.phi
    raw = hwp_test_coefficients(6,0.2)
    identity_beam = prepare_hwp_blm(cp,raw;mueller=Matrix{Float64}(I,4,4))
    @test identity_beam.modes == [:constant]
    ideal = prepare_hwp_blm(cp,raw)
    @test ideal.modes == [:constant,:cos4,:sin4]
    @test ideal.bnmax == 3
    @test ideal.effective_bnmax == 6
    tiny = copy(_HWP_IDEAL_TEST); tiny[1,2]=1e-25
    @test :cos2 in prepare_hwp_blm(cp,raw;mueller=tiny).modes
    for b in ideal.components, l in 0:6
        blocks=blm_slice(b,l); nmax=min(b.bnmax,l)
        for n in -nmax:nmax, s in 1:3
            dual = s == 1 ? 1 : 5-s
            @test blocks[s][-n+nmax+1,1] ≈ (-1.0)^n*conj(blocks[dual][n+nmax+1,1]) atol=1e-15
        end
    end
end

@testset "HWP nominal IQU and detector angle convention" begin
    if CUDA.functional()
        cp=ConvolutionParams(nside_output=4,lmax=6,lstop=6,HWP=true,HWP_rpm=0.)
        g=ring_geometry(cp,7)
        psi=[0.13+0.17i for i in 1:32]; eta=[0.27+0.39i for i in 1:32]
        angle=Angles(cp;pix=g.first_pixel,φ=fill(g.phi_pixels[1],32),θ=fill(g.theta,32),ψ=psi,hwp_angle=eta)
        chi=2 .* psi .+ 4 .* eta
        truth=[1.7,0.3,-0.6]
        signal=truth[1] .+ truth[2].*cos.(chi) .+ truth[3].*sin.(chi)
        vals=reshape(ComplexF64[sum(signal)/32,sum(signal.*cis.(chi))/32],1,1,1,2)
        gpu=CuArray(vals)
        result=RingConvolution{Float64}(7,g.first_pixel,g.pixel_count,[1],[g.first_pixel],gpu)
        @test vec(ring_stokes(result,[angle];rcond=1e-10)) ≈ truth atol=2e-14
        CUDA.unsafe_free!(gpu)
        offsets=[0.23,0.8]; weights=[1.0,2.0]
        vals=zeros(ComplexF64,1,2,1,2)
        for (j,offset) in enumerate(offsets)
            chidet=chi .- 2offset
            d=truth[1] .+ truth[2].*cos.(chidet) .+ truth[3].*sin.(chidet)
            vals[1,j,1,1]=sum(d)/32
            vals[1,j,1,2]=sum(d.*cis.(chi))/32
        end
        grouped=reduce_detector_accumulators(vals,offsets,weights;hwp=true)
        @test vec(grouped_ring_stokes(grouped,[angle],offsets,weights;rcond=1e-10)) ≈ truth atol=2e-14
    else
        @test_skip CUDA.functional()
    end
end

@testset "HWP CPU, direct GPU and GEMM moments" begin
    cp=ConvolutionParams(nside_output=4,lmax=6,lstop=6,bnmax=3,κ=4,HWP=true,HWP_rpm=0.)
    scan=hwp_test_scan(cp)
    buffer=build_ring_scan_buffer(cp,scan.phi,scan.theta,scan.psi;hwp_angle=scan.eta)
    active,angles=make_ring_angles(cp,ring_geometry(cp,7),buffer)
    plan=automatic_scan_plan(ConvolutionParams(cp;HWP=false);beta_order=8,precision=Float64)
    if CUDA.functional()
        for mode in Scanvolver._HWP_MODES
            direct=scan_moments_direct_gpu(plan,angles;hwp_mode=mode,demodulate_hwp=true)
            gemm=scan_moments_gemm_gpu(plan,angles;hwp_mode=mode,demodulate_hwp=true)
            cpu=cat([reshape(scan_moments_cpu(plan,a;hwp_mode=mode,demodulate_hwp=true),2,8,9,7,1)
                     for a in angles]...;dims=5)
            @test Array(direct) ≈ cpu atol=2e-14 rtol=2e-13
            @test Array(gemm) ≈ cpu atol=2e-14 rtol=2e-13
            for (j,a) in enumerate(angles)
                expected=sum(Scanvolver._hwp_modulation(mode,a.η[i])*cis(2a.ψ[i]+4a.η[i]) for i in 1:a.num)/a.num
                @test cpu[2,1,5,4,j] ≈ expected atol=2e-14
            end
            foreach(CUDA.unsafe_free!,(direct,gemm))
        end
        CUDA.unsafe_free!(active.positions)
    else
        @test_skip CUDA.functional()
    end
end

@testset "HWP pipeline against direct Wigner sums" begin
    if CUDA.functional()
        cp=ConvolutionParams(nside_output=4,lmax=6,lstop=6,bnmax=3,κ=12,
                             numofbeam=2,HWP=true,HWP_rpm=0.)
        scan=hwp_test_scan(cp)
        rawsky=hwp_test_coefficients(6,0.9)
        rawbeams=[hwp_test_coefficients(6,0.2),hwp_test_coefficients(6,0.7)]
        sky=prepare_alm(cp,rawsky)
        buffer=build_ring_scan_buffer(cp,scan.phi,scan.theta,scan.psi;hwp_angle=scan.eta)
        cases=[("identity",Matrix{Float64}(I,4,4)),("ideal",_HWP_IDEAL_TEST),
               ("random",_HWP_IDEAL_TEST .+ [0.03sin(0.7i+0.4j) for i in 1:4,j in 1:4])]
        mktempdir() do dir
            for (label,M) in cases
                beams=prepare_hwp_blm(cp,rawbeams;mueller=M)
                tod=hwp_test_direct_tod(cp,rawsky,beams,scan.phi,scan.theta,scan.psi,scan.eta)
                @test maximum(abs,imag.(tod)) < 1e-12maximum(abs,tod)
                expected=fill(NaN,cp.resol.numOfPixels,3,2,1)
                for pixel in unique(scan.pixels)
                    ids=findall(==(pixel),scan.pixels)
                    chi=2 .* scan.psi[ids] .+ 4 .* scan.eta[ids]
                    A=hcat(ones(length(ids)),cos.(chi),sin.(chi))
                    for beam in 1:2
                        expected[pixel,:,beam,1]=A\real.(tod[ids,beam])
                    end
                end
                target=joinpath(dir,label*".h5")
                convolve_to_hdf5(cp,buffer,sky,beams;output_path=target,
                    rings=scan.rings,beam_batch_size=1,beta_order=12,l_tile=3,rcond=1e-10)
                h5open(target,"r") do f
                    actual=read(f["map"])
                    @test actual[unique(scan.pixels),:,:,:] ≈ expected[unique(scan.pixels),:,:,:] atol=3e-11 rtol=3e-10
                    @test read(attributes(f)["hwp_demodulation"]) == "2psi+4eta"
                    @test read(f["hwp_mueller_iquv"]) == M
                end
                if label == "random"
                    offsets=[0.23,0.8]; weights=[1.0,2.0]
                    group_target=joinpath(dir,"grouped.h5")
                    convolve_detector_groups_to_hdf5(cp,buffer,sky,beams;
                        output_path=group_target,rings=scan.rings,beta_order=12,l_tile=2,
                        detector_labels=["a","b"],detector_offsets=offsets,
                        detector_weights=weights,rcond=1e-10)
                    grouped=h5open(f->read(f["map"]),group_target,"r")
                    for pixel in unique(scan.pixels)
                        ids=findall(==(pixel),scan.pixels)
                        chi=2 .* scan.psi[ids] .+ 4 .* scan.eta[ids]
                        normal=zeros(3,3); rhs=zeros(3)
                        for beam in 1:2
                            A=hcat(ones(length(ids)),cos.(chi .- 2offsets[beam]),sin.(chi .- 2offsets[beam]))
                            normal .+= weights[beam] .* (A' * A)
                            rhs .+= weights[beam] .* (A' * real.(tod[ids,beam]))
                        end
                        @test grouped[pixel,:,1,1] ≈ normal\rhs atol=3e-11 rtol=3e-10
                    end
                    single_target=joinpath(dir,"float32_direct.h5")
                    devices=length(CUDA.devices())>=2 && Threads.nthreads()>=2 ? [0,1] : [0]
                    convolve_to_hdf5(cp,buffer,prepare_alm(cp,rawsky;precision=Float32),
                        prepare_hwp_blm(cp,rawbeams;mueller=M,precision=Float32);
                        output_path=single_target,rings=scan.rings,beam_batch_size=2,
                        devices,beta_order=12,l_tile=2,rcond=1e-10,moment_method=:direct)
                    single=h5open(f->read(f["map"]),single_target,"r")
                    @test single[unique(scan.pixels),:,:,:] ≈ expected[unique(scan.pixels),:,:,:] atol=2e-7 rtol=1e-5
                end
                # Optional out-of-repository fixture for an external LBS oracle.
                fixture_dir=get(ENV,"SCANVOLVER_HWP_FIXTURES","")
                if !isempty(fixture_dir)
                    mkpath(fixture_dir)
                    h5open(joinpath(fixture_dir,label*"_fixture.h5"),"w") do f
                        f["sky"]=rawsky; f["beam1"]=rawbeams[1]; f["beam2"]=rawbeams[2]
                        f["mueller"]=M; f["phi"]=scan.phi; f["theta"]=scan.theta
                        f["psi"]=scan.psi; f["eta"]=scan.eta; f["pixels"]=scan.pixels
                        f["tod"]=tod; f["expected_maps"]=expected
                        f["actual_maps"]=h5open(x->read(x["map"]),target,"r")
                        attributes(f)["lmax"]=6; attributes(f)["bnmax"]=3
                    end
                end
            end
        end
    else
        @test_skip CUDA.functional()
    end
end
