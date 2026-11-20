# Uses the HWP Wigner-sum and test-sky helpers in runtests.jl.
using Healpix: pix2angRing
include("data/patanchon_table1.jl")

function patanchon_gaussian_beam(lmax, detector_angle)
    # Polarization and intensity share the same axisymmetric Gaussian window.
    # The TEB coefficients are for a unit-response co-polar detector.
    sigma = deg2rad(23.7 / 60) / sqrt(8log(2))
    window(l) = exp(-0.5l*(l+1)*sigma^2)
    coefficients = zeros(ComplexF64, 3, numberOfAlms(lmax, lmax))
    for l in 0:lmax
        amplitude = window(l) * sqrt((2l+1)/(4pi))
        coefficients[1, l+1] = amplitude
        if l >= 2
            index = 2lmax+l
            polarized = -0.5amplitude * cis(-2detector_angle)
            coefficients[2, index] = polarized
            coefficients[3, index] = im * polarized
        end
    end
    return coefficients, window
end

function patanchon_pixel_scan(cp)
    scan = hwp_test_scan(cp)
    coordinates = [pix2angRing(cp.resol, pixel) for pixel in scan.pixels]
    time = 0.067 .* (0:length(coordinates)-1)
    rho = 2pi * (46/60) .* time .+ 0.29
    return (; phi=last.(coordinates), theta=first.(coordinates), scan.psi,
            scan.pixels, rho, eta=rho .- scan.psi)
end

function patanchon_direct_tod(table, scan, detector_angle, stokes_at_hit)
    detector = [1.0, cos(2detector_angle), sin(2detector_angle), 0.0]
    return [sum(detector .* (
        hwp_test_fourier_response(table, scan.rho[hit] - scan.psi[hit]) *
        hwp_test_stokes_rotation(scan.psi[hit]) * stokes_at_hit(hit)
    )) for hit in eachindex(scan.phi)]
end

@testset "Patanchon Table 1: direct TOD at fixed incidence" begin
    cp = ConvolutionParams(nside_output=4, lmax=6, lstart=2, lstop=6,
                           bnmax=2, κ=12, HWP=true, HWP_rpm=46.)
    scan = patanchon_pixel_scan(cp)
    detector_angle = 0.37
    rawbeam, window = patanchon_gaussian_beam(cp.lmax, detector_angle)
    rawsky = hwp_test_coefficients(cp.lmax, .9)
    rawsky[:, 1:2] .= 0
    rawsky[:, cp.lmax+2] .= 0
    smoothed = [Alm(cp.lmax, cp.lmax, copy(rawsky[stokes,:])) for stokes in 1:3]
    for coeff in smoothed, l in 2:cp.lmax, m in 0:l
        coeff.alm[almIndex(coeff, l, m)] *= window(l)
    end
    map = alm2map(smoothed, cp.nside_output)
    stokes_at_hit(hit) = let pixel = scan.pixels[hit]
        [map.i.pixels[pixel], map.q.pixels[pixel], map.u.pixels[pixel], 0.]
    end
    harmonic2 = (amplitude=PAPER_AMP2, phase=PAPER_PHASE2)
    harmonic4 = (amplitude=PAPER_AMP4, phase=PAPER_PHASE4)
    zero_harmonic = (amplitude=zeros(4,4), phase=zeros(4,4))
    for table in (
        (mueller_0f=PAPER_ZERO, mueller_2f=harmonic2, mueller_4f=harmonic4),
        (mueller_0f=PAPER_ZERO, mueller_2f=zero_harmonic, mueller_4f=zero_harmonic),
        (mueller_0f=zeros(4,4), mueller_2f=harmonic2, mueller_4f=zero_harmonic),
        (mueller_0f=zeros(4,4), mueller_2f=zero_harmonic, mueller_4f=harmonic4),
    )
        expected = patanchon_direct_tod(table, scan, detector_angle, stokes_at_hit)
        prepared = prepare_hwp_blm(cp, rawbeam; table...)
        actual = hwp_test_direct_tod(cp, rawsky, prepared, scan.phi,
                                     scan.theta, scan.psi, scan.eta)[:,1]
        peak = maximum(abs, expected)
        @test peak > 1e-5
        @test maximum(abs, actual-expected) < 1e-9 * peak
    end
end

@testset "Patanchon Table 1: dipole 4f leakage" begin
    # The Gaussian beam needs its high multipoles for the scalar ell=1
    # projection even though the test sky has only a dipole.
    cp = ConvolutionParams(nside_output=4, lmax=2048, lstart=1, lstop=1,
                           bnmax=2, κ=4, HWP=true, HWP_rpm=46.)
    scan = patanchon_pixel_scan(cp)
    detector_angle = 0.37
    rawbeam, window = patanchon_gaussian_beam(cp.lmax, detector_angle)
    rawsky = zeros(ComplexF64, 3, numberOfAlms(cp.lmax, cp.lmax))
    vx, vy, vz = 0.31, -0.22, 0.17
    rawsky[1,2] = vz * sqrt(4pi/3)
    rawsky[1,cp.lmax+2] = -(vx-im*vy)*sqrt(2pi/3)
    amplitude = zeros(4,4)
    amplitude[2:3,1] .= PAPER_AMP4[2:3,1]
    table = (mueller_0f=zeros(4,4),
             mueller_2f=(amplitude=zeros(4,4), phase=zeros(4,4)),
             mueller_4f=(; amplitude, phase=PAPER_PHASE4))
    function dipole_at_hit(hit)
        theta, phi = scan.theta[hit], scan.phi[hit]
        intensity = window(1) * ((vx*cos(phi)+vy*sin(phi))*sin(theta)+vz*cos(theta))
        return [intensity, 0., 0., 0.]
    end
    expected = patanchon_direct_tod(table, scan, detector_angle, dipole_at_hit)
    @test maximum(abs, expected) > 1e-5
    shifted = prepare_hwp_blm(cp, rawbeam; table...)
    corrected = prepare_hwp_blm(cp, rawbeam; table..., lowell_hwp=:full_sum)
    tod(prepared) = hwp_test_direct_tod(cp, rawsky, prepared, scan.phi,
                                       scan.theta, scan.psi, scan.eta)[:,1]
    @test iszero(tod(shifted))
    # The paper's common scalar B_1 and the full polarized beam projection
    # differ slightly at finite width. The measured relative difference is 1.7e-5.
    @test maximum(abs, tod(corrected)-expected) < 3e-5 * maximum(abs, expected)
end
