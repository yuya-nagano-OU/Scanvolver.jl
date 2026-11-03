using Scanvolver
using CUDA
using Test
using LinearAlgebra: I, SymTridiagonal, eigen
using HDF5: h5open, attributes

function lowell_test_d(l, m, n, theta)
    prefactor = sqrt(prod(Float64(factorial(x)) for x in (l+m, l-m, l+n, l-n)))
    return sum((-1.0)^k * prefactor /
        prod(Float64(factorial(x)) for x in (l+m-k, k, n-m+k, l-n-k)) *
        cos(theta/2)^(2l+m-n-2k) * sin(theta/2)^(n-m+2k)
        for k in max(0, m-n):min(l+m, l-n))
end

function lowell_test_beam(lmax, seed)
    raw = zeros(ComplexF64, 3, (lmax+1)*(lmax+2)÷2)
    for m in 0:lmax, l in max(2,m):lmax, s in 1:3
        j = Scanvolver._packed_alm_index(l, m, lmax)
        raw[s,j] = complex(sin(seed+.31s+.47l+.83m),
                          m == 0 ? 0.0 : cos(seed-.27s+.19l-.41m))/(l+1)^2
    end
    raw[1,1] = 1/sqrt(4pi)
    raw[1,2] = .07
    raw[1,lmax+2] = .03+.02im
    return raw
end

function lowell_test_moments(raw, lmax, bnmax)
    # Independent finite-band I,Q,U synthesis and beam-surface integration.
    # No effective-beam or Gaunt helper is used by this oracle.
    nodes = eigen(SymTridiagonal(zeros(32), [j/sqrt(4j^2-1) for j in 1:31]))
    mean, moment = zeros(3), zeros(3,3)
    for j in eachindex(nodes.values), i in 0:39
        theta, phi = acos(nodes.values[j]), 2pi*i/40
        intensity, p = 0.0im, 0.0im
        for l in 0:lmax, m in -min(bnmax,l):min(bnmax,l)
            index = Scanvolver._packed_alm_index(l, abs(m), lmax)
            b0 = raw[1,index]
            plus, minus = -(raw[2,index]+im*raw[3,index]), -(raw[2,index]-im*raw[3,index])
            if m < 0
                b0 = (-1.0)^m*conj(b0)
                plus = (-1.0)^m*conj(minus)
            end
            norm = sqrt((2l+1)/(4pi))*cis(m*phi)
            intensity += b0*norm*lowell_test_d(l,m,0,theta)
            l >= 2 && (p += plus*norm*lowell_test_d(l,m,-2,theta))
        end
        p *= cis(2phi) # spherical polarization -> Ludwig-3.
        response = [real(intensity), real(p), imag(p)]
        xyz = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
        area = 2*nodes.vectors[1,j]^2 * (2pi/40)
        mean .+= area .* response
        moment .+= area .* (response*xyz')
    end
    return mean, moment
end

function lowell_test_reference(raw, cp, mono, dipole, matrix, phi, theta, psi, eta)
    mean, moment = lowell_test_moments(raw, cp.lmax, cp.bnmax)
    output = zeros(length(phi))
    vx, vy, vz = dipole
    for i in eachindex(phi)
        dx = cos(theta[i])*(cos(phi[i])*vx+sin(phi[i])*vy)-sin(theta[i])*vz
        dy = -sin(phi[i])*vx+cos(phi[i])*vy
        dz = sin(theta[i])*(cos(phi[i])*vx+sin(phi[i])*vy)+cos(theta[i])*vz
        local_d = [cos(psi[i])*dx+sin(psi[i])*dy,
                   -sin(psi[i])*dx+cos(psi[i])*dy, dz]
        integrals = mono*mean + moment*local_d
        c, s = cos(2eta[i]), sin(2eta[i])
        output[i] = integrals[1]*matrix[1,1] +
            integrals[2]*(c*matrix[2,1]-s*matrix[3,1]) +
            integrals[3]*(s*matrix[2,1]+c*matrix[3,1])
    end
    return output
end

function lowell_test_tod(cp, rawsky, prepared, phi, theta, psi, eta)
    sky = prepare_alm(cp, rawsky)
    output = zeros(ComplexF64, length(phi), cp.numofbeam)
    for l in cp.lstart:min(1, cp.lstop), i in eachindex(phi)
        a = alm_slice(sky, l)[1]
        for (mode, component) in zip(prepared.modes, prepared.components)
            b = blm_slice(component, l)[1]
            for m in -l:l, n in -l:l, beam in 1:cp.numofbeam
                output[i,beam] += a[1,m+l+1]*b[n+l+1,beam] *
                    lowell_test_d(l,m,n,theta[i])*cis(m*phi[i]+n*psi[i]) *
                    Scanvolver._hwp_modulation(mode, eta[i])
            end
        end
    end
    return output
end

function lowell_test_same(a, b; first_l=a.lstart)
    return a.modes == b.modes && all(
        blm_slice(x,l) == blm_slice(y,l)
        for (x,y) in zip(a.components,b.components) for l in first_l:a.lstop)
end

@testset "Low-ell HWP against independent real-space TOD and maps" begin
    cp = ConvolutionParams(nside_output=4,lmax=6,lstop=6,bnmax=3,κ=12,
                           numofbeam=2,HWP=true,HWP_rpm=0.)
    mono, dipole = .7, [.31,-.22,.17]
    sky = zeros(ComplexF64,3,28)
    sky[1,1] = mono*sqrt(4pi)
    sky[1,2] = dipole[3]*sqrt(4pi/3)
    sky[1,8] = -(dipole[1]-im*dipole[2])*sqrt(2pi/3)
    beams = [lowell_test_beam(6,.2), lowell_test_beam(6,.7)]
    beams[2][1,:] .= 0 # P-only diagnostic in the same multi-beam call.
    # A rotated passive HWP exercises both M_QI and M_UI.
    c, s = cos(.46), sin(.46)
    rotation = [1. 0 0 0; 0 c s 0; 0 -s c 0; 0 0 0 1]
    g = sqrt(.99*.97)
    matrix = rotation' * [.98 .01 0 0; .01 .98 0 0; 0 0 -g 0; 0 0 0 -g] * rotation
    prepared = prepare_hwp_blm(cp,beams;mueller=matrix,lowell_hwp=:full_sum)
    phi, theta, psi, eta = Float64[], Float64[], Float64[], Float64[]
    pixels = Int[]
    for ring in (4,7,11), position in (1,3), hit in 1:24
        geometry = ring_geometry(cp,ring)
        push!(phi,geometry.phi_pixels[position]); push!(theta,geometry.theta)
        push!(psi,.43hit+.17position); push!(eta,.71hit+.23position+.13ring)
        push!(pixels,geometry.first_pixel+position-1)
    end
    actual = lowell_test_tod(cp,sky,prepared,phi,theta,psi,eta)
    reference = hcat([lowell_test_reference(beam,cp,mono,dipole,matrix,phi,theta,psi,eta) for beam in beams]...)
    @test maximum(abs,imag.(actual)) < 1e-14
    @test real.(actual) ≈ reference atol=2e-12 rtol=2e-11
    @test maximum(abs,real.(actual[:,2])) > 1e-5
    ids = unique(pixels)
    angles = Angles[]
    moments = zeros(ComplexF64,1,2,length(ids),2)
    expected = zeros(length(ids),3,2)
    for (j,pixel) in enumerate(ids)
        keep = findall(==(pixel),pixels)
        push!(angles,Angles(cp;pix=pixel,φ=phi[keep],θ=theta[keep],ψ=psi[keep],hwp_angle=eta[keep]))
        chi = 2psi[keep]+4eta[keep]
        design = hcat(ones(length(keep)),cos.(chi),sin.(chi))
        expected[j,:,:] = design \ reference[keep,:]
        for beam in 1:2
            moments[1,beam,j,1] = sum(actual[keep,beam])/length(keep)
            moments[1,beam,j,2] = sum(actual[keep,beam].*cis.(chi))/length(keep)
        end
    end
    mapped = grouped_ring_stokes(moments,angles,[0.],[1.];rcond=1e-10)
    @test mapped[1,:,:,:] ≈ permutedims(expected,(3,2,1)) atol=2e-12 rtol=2e-11
    if CUDA.functional()
        buffer = build_ring_scan_buffer(cp,phi,theta,psi;hwp_angle=eta)
        mktempdir() do dir
            target = joinpath(dir,"lowell.h5")
            convolve_to_hdf5(cp,buffer,prepare_alm(cp,sky),prepared;output_path=target,
                rings=[4,7,11],beam_batch_size=1,beta_order=12,l_tile=3,rcond=1e-10)
            h5open(target,"r") do file
                @test read(file["map"])[ids,:,:,1] ≈ expected atol=3e-11 rtol=3e-10
                @test read(attributes(file)["hwp_lowell"]) == "full_sum"
            end
        end
    else
        @test_skip CUDA.functional()
    end
end

@testset "Low-ell scope, layout, truncation and precision" begin
    cp = ConvolutionParams(nside_output=4,lmax=6,lstop=6,bnmax=3,HWP=true,HWP_rpm=0.)
    raw = lowell_test_beam(6,.2)
    matrix = [.98 .01 0 0; .01 .98 0 0; 0 0 -.97 0; 0 0 0 -.97]
    shifted = prepare_hwp_blm(cp,raw;mueller=matrix)
    full = prepare_hwp_blm(cp,raw;mueller=matrix,lowell_hwp=:full_sum)
    @test lowell_test_same(shifted,prepare_hwp_blm(cp,raw;mueller=matrix,lowell_hwp=:shifted))
    @test lowell_test_same(shifted,full;first_l=2)
    @test all(all(iszero,blm_slice(component,l)[s]) for component in full.components for l in 0:1 for s in 2:3)
    for M in (Matrix{Float64}(I,4,4), [1. 0 .02 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1])
        @test lowell_test_same(prepare_hwp_blm(cp,raw;mueller=M),prepare_hwp_blm(cp,raw;mueller=M,lowell_hwp=:full_sum))
    end
    limited = prepare_hwp_blm(ConvolutionParams(cp;lstop=1),raw;mueller=matrix,lowell_hwp=:full_sum)
    @test all(blm_slice(a,l)==blm_slice(b,l) for (a,b) in zip(full.components,limited.components) for l in 0:1)
    spin = copy(raw)
    spin[2,:] = -(raw[2,:]+im*raw[3,:]); spin[3,:] = -(raw[2,:]-im*raw[3,:])
    @test lowell_test_same(full,prepare_hwp_blm(cp,spin;mueller=matrix,input_basis=:spin,lowell_hwp=:full_sum))
    single = prepare_hwp_blm(cp,raw;mueller=matrix,precision=Float32,lowell_hwp=:full_sum)
    @test all(isapprox(blm_slice(a,l)[1],blm_slice(b,l)[1];atol=1e-8,rtol=1e-6)
              for (a,b) in zip(full.components,single.components) for l in 0:6)
    @test only(beam_batches(full,1)).blm.lowell_hwp === :full_sum
    @test_throws ArgumentError prepare_hwp_blm(cp,raw;lowell_hwp=:invalid)
    @test Scanvolver._hwp_lowell_weights(0,0,2,4)[3] ≈ sqrt(5)/3 atol=1e-15
end
