const _HWP_MODES = (:constant, :cos2, :sin2, :cos4, :sin4)
const _HWP_SIGMA = (0, 2, -2)
const _HWP_SCALES = (1.0, inv(sqrt(2.0)), inv(sqrt(2.0)))
const _HWP_IDEAL_MUELLER = [1.0 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]

const _HWPHarmonicTable = NamedTuple{(:amplitude, :phase),Tuple{Matrix{Float64},Matrix{Float64}}}
const _HWPFourierTable = NamedTuple{(:mueller_0f, :mueller_2f, :mueller_4f),
                                    Tuple{Matrix{Float64},_HWPHarmonicTable,_HWPHarmonicTable}}

"""Five real Fourier beams of an HWP Mueller response (V=0 inputs).

The input beam cutoff `bnmax` is kept separate from `effective_bnmax`.
Each component is already conjugated, just like `PreparedBlm`. HWP angles
are measured in the same focal-plane frame as the supplied beam coefficients.

Exactly one of `mueller` (a 4x4 HWP-frame matrix rotated by the HWP angle) and
`mueller_fourier` (the 0f, 2f and 4f focal-plane tables) is set; the other is
`nothing`. They record the input of `prepare_hwp_blm`.
"""
struct PreparedHWPBlm{T<:_HARMONIC_PRECISIONS}
    components::Vector{PreparedBlm{T}}
    modes::Vector{Symbol}
    mueller::Union{Nothing,Matrix{Float64}}
    mueller_fourier::Union{Nothing,_HWPFourierTable}
    lowell_hwp::Symbol
    lstart::Int
    lstop::Int
    lmax::Int
    bnmax::Int
    effective_bnmax::Int
    numofbeam::Int
end

function _hwp_coherence(mueller)
    size(mueller) == (4, 4) && isreal(mueller) ||
        throw(ArgumentError("mueller must be a real 4 by 4 IQUV matrix"))
    M = Float64.(real.(mueller))
    all(isfinite, M) || throw(ArgumentError("mueller must be finite"))
    C = zeros(ComplexF64, 3, 3)
    C[1, 1] = M[1, 1]
    C[1, 2] = (M[1, 2] - im * M[1, 3]) / sqrt(2.0)
    C[1, 3] = conj(C[1, 2])
    C[2, 1] = (M[2, 1] + im * M[3, 1]) / sqrt(2.0)
    C[3, 1] = conj(C[2, 1])
    # Form the symmetry sums before normalization, preserving exact zeros.
    C[2, 2] = ((M[2, 2] + M[3, 3]) + im * (M[3, 2] - M[2, 3])) / 2
    C[2, 3] = ((M[2, 2] - M[3, 3]) + im * (M[3, 2] + M[2, 3])) / 2
    C[3, 2] = conj(C[2, 3])
    C[3, 3] = conj(C[2, 2])
    return M, C
end

# Complex coefficients Γ_j of the five modulation functions
# f = (1, cos2η, sin2η, cos4η, sin4η), with C(η) = Σ_j Γ_j f_j(η).

"""Distribute a rotated fixed matrix by its harmonic t=sigma[X]-sigma[Y]."""
function _rotating_channels(C)
    channels = ntuple(_ -> zeros(ComplexF64, 3, 3), length(_HWP_MODES))
    for X in 1:3, Y in 1:3
        iszero(C[X, Y]) && continue
        t = _HWP_SIGMA[X] - _HWP_SIGMA[Y]
        if t == 0
            channels[1][X, Y] = C[X, Y]
        else
            cosine = abs(t) == 2 ? 2 : 4
            channels[cosine][X, Y] = C[X, Y]
            channels[cosine + 1][X, Y] = (t > 0 ? im : -im) * C[X, Y]
        end
    end
    return channels
end

function _fourier_matrix(value, name)
    value isa AbstractMatrix && size(value) == (4, 4) && isreal(value) ||
        throw(ArgumentError("$name must be a real 4 by 4 IQUV matrix"))
    matrix = Float64.(real.(value))
    all(isfinite, matrix) || throw(ArgumentError("$name must be finite"))
    return matrix
end

function _fourier_harmonic(value, name)
    value === nothing && return (amplitude=zeros(4, 4), phase=zeros(4, 4))
    value isa NamedTuple && length(value) == 2 &&
        haskey(value, :amplitude) && haskey(value, :phase) ||
        throw(ArgumentError("$name must be a named tuple (amplitude=..., phase=...)"))
    return (amplitude=_fourier_matrix(value.amplitude, "$name.amplitude"),
            phase=_fourier_matrix(value.phase, "$name.phase"))
end

"""Validate the 0f/2f/4f tables and return them with the complex channels.

Element by element, M(η) = M0f + Σ_{n=2,4} A_n cos(nη + φ_n). The cosine and
sine channels are A_n cos(φ_n) and -A_n sin(φ_n).
"""
function _fourier_channels(mueller_0f, mueller_2f, mueller_4f)
    table = (mueller_0f=_fourier_matrix(mueller_0f, "mueller_0f"),
             mueller_2f=_fourier_harmonic(mueller_2f, "mueller_2f"),
             mueller_4f=_fourier_harmonic(mueller_4f, "mueller_4f"))
    real_channels = (table.mueller_0f,
        table.mueller_2f.amplitude .* cos.(table.mueller_2f.phase),
        .-(table.mueller_2f.amplitude .* sin.(table.mueller_2f.phase)),
        table.mueller_4f.amplitude .* cos.(table.mueller_4f.phase),
        .-(table.mueller_4f.amplitude .* sin.(table.mueller_4f.phase)))
    return table, map(matrix -> last(_hwp_coherence(matrix)), real_channels)
end

"""
    prepare_hwp_blm(cp, blms; mueller=diag(1,1,-1,-1), input_basis=:teb,
                    precision=Float64, lowell_hwp=:shifted)
    prepare_hwp_blm(cp, blms; mueller_0f, mueller_2f=nothing, mueller_4f=nothing,
                    input_basis=:teb, precision=Float64, lowell_hwp=:shifted)

Prepare an HWP Mueller element in front of asymmetric detector beams.
Inputs are TEB (or spin 0,+2,-2); sky V and beam V are zero. Matrices are real
4x4 in IQUV order, rows output and columns input; their V row and column do
not contribute. This implements the shifted-beam narrow-beam, fundamental
spatial Mueller-mode approximation also used by LBS/SBM.

The response can be given in one of two forms:

- `mueller`: a fixed HWP-frame matrix M, rotated by the HWP angle η as
  R(-η) M R(η). This is the default, with the ideal HWP.
- `mueller_0f`, `mueller_2f`, `mueller_4f`: the focal-plane response in the
  form of Patanchon et al. (2023), Table 1,
  `M(η) = M0f + Σ_{n=2,4} A_n .* cos.(n*η .+ φ_n)`, element by element.
  `mueller_0f` is the signed constant matrix; `mueller_2f` and `mueller_4f`
  are `(amplitude=A_n, phase=φ_n)` with phases in radians, and an omitted
  harmonic is zero. This response already contains the HWP rotation; no
  further rotation is applied. η is the mechanical HWP angle relative to the
  focal plane (the paper's ρ-ψ), and the beam coefficients use the same frame.

In the normalized complex basis, the response is C(η)=Σ_j Γ_j f_j(η) with
f=(1, cos2η, sin2η, cos4η, sin4η) and Γ_j=T*M_j*T†. An entry Γ_j[X,Y] moves
the beam order by t=sigma[X]-sigma[Y], sigma=(0,2,-2), whatever its HWP
harmonic. Its conjugated effective beam at Wigner order n is
(scale[X]/scale[Y])*Γ_j[X,Y]*conj(b[X,n-t]), so no second conjugation is
applied to the prepared physical beam. For a rotated fixed matrix the HWP
harmonic equals t. Only exactly zero components are omitted; weak 2f terms
are retained.

With `lowell_hwp=:full_sum`, retain the beam multipole sum for the scalar
I-input ell=0,1 terms proportional to the M_QI and M_UI entries of every
channel. The Mueller matrix is uniform in beam-centred Ludwig-3 coordinates.
The sum uses polarized input beam multipoles 2:lmax (even when lstop<lmax),
truncated at bnmax. All other terms retain the shifted-beam approximation.
The default `:shifted` preserves the original results, and spin-2 ell<2
stays zero.
"""
function prepare_hwp_blm(cp::ConvolutionParams, blms::AbstractVector;
                         mueller=nothing, mueller_0f=nothing,
                         mueller_2f=nothing, mueller_4f=nothing,
                         input_basis::Symbol=:teb, precision::Type=Float64,
                         lowell_hwp::Symbol=:shifted)
    cp.HWP || throw(ArgumentError("prepare_hwp_blm requires HWP=true"))
    lowell_hwp in (:shifted, :full_sum) ||
        throw(ArgumentError("lowell_hwp must be :shifted or :full_sum"))
    if mueller_0f === nothing && mueller_2f === nothing && mueller_4f === nothing
        M, C = _hwp_coherence(mueller === nothing ? _HWP_IDEAL_MUELLER : mueller)
        channels = _rotating_channels(C)
        table = nothing
    else
        mueller === nothing || throw(ArgumentError(
            "pass either mueller or mueller_0f/mueller_2f/mueller_4f, not both"))
        mueller_0f === nothing && throw(ArgumentError(
            "mueller_0f is required with mueller_2f or mueller_4f"))
        M = nothing
        table, channels = _fourier_channels(mueller_0f, mueller_2f, mueller_4f)
    end
    physical_cp = ConvolutionParams(cp; HWP=false)
    physical = prepare_blm(physical_cp, blms; input_basis, precision)
    T = precision
    lowell = if lowell_hwp === :full_sum && cp.lstart <= 1 && cp.lmax >= 2 &&
                any(channel -> any(!iszero, channel[2:3, 1]), channels)
        _prepare_hwp_lowell(cp, blms, input_basis, channels, T)
    else
        nothing
    end
    effective_bnmax = min(cp.bnmax + 4, cp.lmax)
    slices = [Vector{NTuple{3,Matrix{Complex{T}}}}() for _ in _HWP_MODES]
    for l in cp.lstart:cp.lstop
        nmax = min(effective_bnmax, l)
        physical_nmax = min(cp.bnmax, l)
        source = blm_slice(physical, l)
        blocks = [ntuple(_ -> zeros(Complex{T}, 2nmax + 1, cp.numofbeam), 3)
                  for _ in _HWP_MODES]
        for X in 1:3, Y in 1:3
            ((X > 1 || Y > 1) && l < 2) && continue
            t = _HWP_SIGMA[X] - _HWP_SIGMA[Y]
            ratio = _HWP_SCALES[X] / _HWP_SCALES[Y]
            for j in eachindex(_HWP_MODES)
                iszero(channels[j][X, Y]) && continue
                factor = ratio * channels[j][X, Y]
                for n in -nmax:nmax
                    input_order = n - t
                    abs(input_order) <= physical_nmax || continue
                    row = n + nmax + 1
                    for beam in 1:cp.numofbeam
                        blocks[j][Y][row, beam] += Complex{T}(
                            factor * source[X][input_order + physical_nmax + 1, beam])
                    end
                end
            end
        end
        if lowell !== nothing && l < 2
            for j in eachindex(blocks)
                blocks[j][1] .+= lowell[l][j]
            end
        end
        for j in eachindex(slices)
            push!(slices[j], blocks[j])
        end
    end
    active = [j for j in eachindex(slices)
              if any(block -> any(x -> any(!iszero, x), block), slices[j])]
    isempty(active) && push!(active, 1)
    components = [PreparedBlm{T}(slices[j], cp.lstart, cp.lstop, cp.lmax,
                                effective_bnmax, cp.numofbeam) for j in active]
    return PreparedHWPBlm{T}(components, collect(_HWP_MODES)[active], M, table, lowell_hwp,
                            cp.lstart, cp.lstop, cp.lmax, cp.bnmax,
                            effective_bnmax, cp.numofbeam)
end

prepare_hwp_blm(cp::ConvolutionParams, blm::AbstractMatrix; kwargs...) =
    prepare_hwp_blm(cp, [blm]; kwargs...)

function beam_batches(blm::PreparedHWPBlm{T}, batch_size::Integer) where {T}
    batches = [beam_batches(component, batch_size) for component in blm.components]
    return [(indices=batches[1][j].indices,
             blm=PreparedHWPBlm{T}([batch[j].blm for batch in batches], copy(blm.modes),
                  deepcopy(blm.mueller), deepcopy(blm.mueller_fourier), blm.lowell_hwp,
                  blm.lstart, blm.lstop, blm.lmax, blm.bnmax,
                  blm.effective_bnmax, length(batches[1][j].indices)))
            for j in eachindex(batches[1])]
end

_geometry_params(cp, ::PreparedBlm) = cp
_geometry_params(cp, blm::PreparedHWPBlm) =
    ConvolutionParams(cp; HWP=false, bnmax=blm.effective_bnmax)
_beam_component(blm::PreparedBlm, j) = blm
_beam_component(blm::PreparedHWPBlm, j) = blm.components[j]

@inline function _hwp_modulation(mode::Symbol, eta::Real)
    mode === :constant && return one(eta)
    mode === :cos2 && return cos(2eta)
    mode === :sin2 && return sin(2eta)
    mode === :cos4 && return cos(4eta)
    mode === :sin4 && return sin(4eta)
    throw(ArgumentError("unknown HWP mode: $mode"))
end

function _check_hwp_moment_inputs(angles, mode, demodulate_hwp)
    mode in _HWP_MODES || throw(ArgumentError("unknown HWP mode: $mode"))
    if demodulate_hwp || mode !== :constant
        for angle in angles
            length(angle.η) == angle.num && all(isfinite, angle.η) ||
                throw(ArgumentError("HWP scan moments require one finite eta per hit"))
        end
    end
end

@inline _sample_modulation(::Nothing, i, ::Type{T}) where {T} = one(T)
@inline _sample_modulation(weights, i, ::Type{T}) where {T} = @inbounds weights[i]

@inline _has_hwp_angles(angle) = hasproperty(angle, :η) && !isempty(angle.η)

@inline function _nominal_pol_angle(angle, i)
    return _has_hwp_angles(angle) ? angle.ψ[i] + 2angle.η[i] : angle.ψ[i]
end

function _pipeline_moments(plan, angles, ::PreparedBlm; kwargs...)
    return [scan_moments_gpu(plan, angles; kwargs...)]
end

function _pipeline_moments(plan, angles, blm::PreparedHWPBlm; kwargs...)
    moments = CuArray{Complex{typeof(plan.beta_max)},5}[]
    try
        for mode in blm.modes
            push!(moments, scan_moments_gpu(plan, angles; hwp_mode=mode,
                                           demodulate_hwp=true, kwargs...))
        end
        return moments
    catch
        foreach(CUDA.unsafe_free!, moments)
        rethrow()
    end
end
