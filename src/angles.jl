"""
    Angles(cp; pix, φ, θ, ψ, time_array=nothing, hwp_angle=nothing, hwp_phase_rad=0, eps=1e-12)

Per-sample local Euler angles for one HEALPix pixel. The pointing vectors
`φ`, `θ`, and `ψ` must have equal lengths. With `cp.HWP`, provide measured
`hwp_angle` samples or absolute `time_array` samples. Generated phases include
`hwp_phase_rad`; explicit angles are used as supplied modulo 2π.
"""
mutable struct Angles
    θ_pix::Float64
    φ_pix::Float64
    ψ::Vector{Float64}
    α::Vector{Float64}
    β::Vector{Float64}
    γ::Vector{Float64}
    time_array::Vector{Float64}
    η::Vector{Float64}
    num::Int
end

function Angles(
    cp::ConvolutionParams;
    pix::Int,
    φ::Vector{T},
    θ::Vector{T},
    ψ::Vector{T},
    time_array::Union{Nothing, AbstractVector{<:Real}}=nothing,
    hwp_angle::Union{Nothing, AbstractVector{<:Real}}=nothing,
    hwp_phase_rad::Real=0.0,
    eps::T=1e-12,
) where {T<:AbstractFloat}
    sample_count = length(φ)
    length(θ) == sample_count || throw(ArgumentError(
        "θ must have length $sample_count, got $(length(θ))",
    ))
    length(ψ) == sample_count || throw(ArgumentError(
        "ψ must have length $sample_count, got $(length(ψ))",
    ))

    if time_array === nothing
        cp.HWP && hwp_angle === nothing && throw(ArgumentError(
            "time_array must be provided when HWP=true",
        ))
        times = Float64[]
    else
        length(time_array) == sample_count || throw(ArgumentError(
            "time_array length must match φ, θ, and ψ",
        ))
        times = Float64.(time_array)
        all(isfinite, times) || throw(ArgumentError(
            "time_array must contain only finite values",
        ))
    end

    isfinite(hwp_phase_rad) || throw(ArgumentError("hwp_phase_rad must be finite"))
    if hwp_angle !== nothing
        cp.HWP || throw(ArgumentError("hwp_angle requires HWP=true"))
        length(hwp_angle) == sample_count && all(isfinite, hwp_angle) ||
            throw(ArgumentError("hwp_angle requires one finite angle per hit"))
        η = mod.(Float64.(hwp_angle), 2π)
    elseif cp.HWP
        angular_velocity = 2π * cp.HWP_rpm / 60
        η = mod.(angular_velocity .* times .+ hwp_phase_rad, 2π)
    else
        η = Float64[]
    end

    θ_pix, φ_pix = pix2angRing(cp.resol, pix)
    dphi = φ .- φ_pix
    dtheta = θ .- θ_pix
    α, β, γ = split_rotation_vectors(
        φ_pix,
        θ_pix,
        dphi,
        dtheta,
        ψ;
        eps=eps,
    )

    return Angles(
        θ_pix,
        φ_pix,
        Float64.(ψ),
        α,
        β,
        γ,
        times,
        η,
        sample_count,
    )
end
