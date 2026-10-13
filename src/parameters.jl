"""
    ConvolutionParams(; kwargs...)

Configuration shared by harmonic preparation, scan processing, convolution,
and mapmaking. The multipole range must satisfy
`0 <= lstart <= lstop <= lmax`; `numofsky` and `numofbeam` set the batch axes,
`bnmax` limits beam azimuthal modes, and `κ` controls the scan-moment band.

The current GPU convolution path supports `HWP=false`. When `HWP=true`,
`HWP_rpm` is required and is used while constructing `Angles`.
"""
struct ConvolutionParams{I<:Int, B<:Bool}
    nside_output::I
    resol::Resolution
    lstart::I
    lstop::I
    HWP::B
    HWP_rpm::Union{Nothing, Float64}
    numofsky::I
    lmax::I
    Mmax_max::I
    numofbeam::I
    bnmax::I
    κ::I
    noise::B
    noise_sigma::Union{Nothing, Float64}
end

function ConvolutionParams(;
    nside_output::Int=128,
    lstart::Int=0,
    lstop::Int=3 * nside_output - 1,
    HWP::Bool=false,
    HWP_rpm::Union{Nothing, Real}=nothing,
    numofsky::Int=1,
    lmax::Int=3 * nside_output - 1,
    numofbeam::Int=1,
    bnmax::Int=2,
    κ::Int=20,
    noise::Bool=false,
    noise_sigma::Union{Nothing, Real}=nothing,
)
    nside_output > 0 || throw(ArgumentError(
        "nside_output must be positive, got $nside_output",
    ))
    0 <= lstart <= lstop <= lmax || throw(ArgumentError(
        "multipoles must satisfy 0 <= lstart <= lstop <= lmax, " *
        "got $lstart <= $lstop <= $lmax",
    ))
    numofsky > 0 || throw(ArgumentError(
        "numofsky must be positive, got $numofsky",
    ))
    numofbeam > 0 || throw(ArgumentError(
        "numofbeam must be positive, got $numofbeam",
    ))
    0 <= bnmax <= lmax || throw(ArgumentError(
        "bnmax must satisfy 0 <= bnmax <= lmax, " *
        "got bnmax=$bnmax and lmax=$lmax",
    ))
    κ >= 0 || throw(ArgumentError(
        "κ must be non-negative, got $κ",
    ))
    Mmax_max = min(κ + bnmax, lmax)
    resol = Resolution(nside_output)

    if HWP
        HWP_rpm === nothing && throw(ArgumentError(
            "HWP_rpm must be provided when HWP=true",
        ))
        HWP_rpm = Float64(HWP_rpm)
        isfinite(HWP_rpm) ||
            throw(ArgumentError("HWP_rpm must be finite"))
    else
        HWP_rpm = nothing
    end

    if noise_sigma !== nothing
        noise_sigma = Float64(noise_sigma)
        isfinite(noise_sigma) ||
            throw(ArgumentError("noise_sigma must be finite"))
        noise_sigma >= 0 ||
            throw(ArgumentError("noise_sigma must be non-negative"))
        noise = true
    elseif noise
        throw(ArgumentError(
            "noise_sigma must be provided when noise=true",
        ))
    end

    return ConvolutionParams(
        nside_output,
        resol,
        lstart,
        lstop,
        HWP,
        HWP_rpm,
        numofsky,
        lmax,
        Mmax_max,
        numofbeam,
        bnmax,
        κ,
        noise,
        noise_sigma,
    )
end

function ConvolutionParams(
    cp::ConvolutionParams;
    nside_output::Int=cp.nside_output,
    lstart::Int=cp.lstart,
    lstop::Int=cp.lstop,
    HWP::Bool=cp.HWP,
    HWP_rpm::Union{Nothing, Real}=cp.HWP_rpm,
    numofsky::Int=cp.numofsky,
    lmax::Int=cp.lmax,
    numofbeam::Int=cp.numofbeam,
    bnmax::Int=cp.bnmax,
    κ::Int=cp.κ,
    noise::Bool=cp.noise,
    noise_sigma::Union{Nothing, Real}=cp.noise_sigma,
)
    return ConvolutionParams(
        ;
        nside_output,
        lstart,
        lstop,
        HWP,
        HWP_rpm,
        numofsky,
        lmax,
        numofbeam,
        bnmax,
        κ,
        noise,
        noise_sigma,
    )
end

function Base.show(
    io::IO,
    ::MIME"text/plain",
    cp::ConvolutionParams,
)
    println(io, "nside_output: ", cp.nside_output)
    println(io, "resol: ", cp.resol)
    println(io, "lstart: ", cp.lstart)
    println(io, "lstop: ", cp.lstop)
    println(io, "HWP: ", cp.HWP)
    println(io, "HWP_rpm: ", cp.HWP_rpm)
    println(io, "numofsky: ", cp.numofsky)
    println(io, "lmax: ", cp.lmax)
    println(io, "numofbeam: ", cp.numofbeam)
    println(io, "bnmax: ", cp.bnmax)
    println(io, "κ: ", cp.κ)
    println(io, "noise: ", cp.noise)
    print(io, "noise_sigma: ", cp.noise_sigma)
end

function Base.show(io::IO, cp::ConvolutionParams)
    print(
        io,
        "ConvolutionParams(nside=$(cp.nside_output), ",
        "ℓ=$(cp.lstart):$(cp.lstop)/$(cp.lmax), ",
        "bnmax=$(cp.bnmax), κ=$(cp.κ))",
    )
end

"""
    show_cp(cp)

Print the fields of `cp` in a readable multi-line form.
"""
function show_cp(cp::ConvolutionParams)
    show(stdout, MIME"text/plain"(), cp)
    println()
    return nothing
end
