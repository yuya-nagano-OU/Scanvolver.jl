const _HWP_MODES = (:constant, :cos2, :sin2, :cos4, :sin4)

"""Five real Fourier beams of a rotating, real IQUV Mueller matrix (V=0 inputs).

The input beam cutoff `bnmax` is kept separate from `effective_bnmax`.
Each component is already conjugated, just like `PreparedBlm`. HWP angles
are measured in the same focal-plane frame as the supplied beam coefficients.
"""
struct PreparedHWPBlm{T<:_HARMONIC_PRECISIONS}
    components::Vector{PreparedBlm{T}}
    modes::Vector{Symbol}
    mueller::Matrix{Float64}
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

"""
    prepare_hwp_blm(cp, blms; mueller=diag(1,1,-1,-1), input_basis=:teb,
                    precision=Float64, lowell_hwp=:shifted)

Prepare a rotating Mueller element in front of asymmetric detector beams.
Inputs are TEB (or spin 0,+2,-2); sky V and beam V are zero. The real 4x4
matrix is in IQUV order. This implements the shifted-beam narrow-beam,
fundamental spatial Mueller-mode approximation also used by LBS/SBM.

In the normalized complex basis C=T*M*T†, an entry C[X,Y] has harmonic
t=sigma[X]-sigma[Y], sigma=(0,2,-2). Its conjugated effective beam at Wigner
order n is (scale[X]/scale[Y])*C[X,Y]*conj(b[X,n-t]). Thus no second
conjugation is applied to the prepared physical beam. Signed modes are
combined analytically into 1, cos(2η), sin(2η), cos(4η), sin(4η).
Only exactly zero components are omitted; weak 2f terms are retained.

With `lowell_hwp=:full_sum`, retain the beam multipole sum for the scalar
I-input ell=0,1 2f terms proportional to M_QI and M_UI. The Mueller matrix
is uniform in beam-centred Ludwig-3 coordinates. The sum uses polarized
input beam multipoles 2:lmax (even when lstop<lmax), truncated at bnmax.
All other terms retain the shifted-beam approximation. The default
`:shifted` preserves the original results, and spin-2 ell<2 stays zero.
"""
function prepare_hwp_blm(cp::ConvolutionParams, blms::AbstractVector;
                         mueller=[1.0 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1],
                         input_basis::Symbol=:teb, precision::Type=Float64,
                         lowell_hwp::Symbol=:shifted)
    cp.HWP || throw(ArgumentError("prepare_hwp_blm requires HWP=true"))
    lowell_hwp in (:shifted, :full_sum) ||
        throw(ArgumentError("lowell_hwp must be :shifted or :full_sum"))
    M, C = _hwp_coherence(mueller)
    physical_cp = ConvolutionParams(cp; HWP=false)
    physical = prepare_blm(physical_cp, blms; input_basis, precision)
    T = precision
    lowell = if lowell_hwp === :full_sum && cp.lstart <= 1 && cp.lmax >= 2 &&
                any(!iszero, C[2:3, 1])
        _prepare_hwp_lowell(cp, blms, input_basis, C, T)
    else
        nothing
    end
    effective_bnmax = min(cp.bnmax + 4, cp.lmax)
    sigma = (0, 2, -2)
    scales = (1.0, inv(sqrt(2.0)), inv(sqrt(2.0)))
    slices = [Vector{NTuple{3,Matrix{Complex{T}}}}() for _ in _HWP_MODES]
    for l in cp.lstart:cp.lstop
        nmax = min(effective_bnmax, l)
        physical_nmax = min(cp.bnmax, l)
        source = blm_slice(physical, l)
        blocks = [ntuple(_ -> zeros(Complex{T}, 2nmax + 1, cp.numofbeam), 3)
                  for _ in _HWP_MODES]
        for X in 1:3, Y in 1:3
            ((X > 1 || Y > 1) && l < 2) && continue
            iszero(C[X, Y]) && continue
            t = sigma[X] - sigma[Y]
            factor = scales[X] / scales[Y] * C[X, Y]
            for n in -nmax:nmax
                input_order = n - t
                abs(input_order) <= physical_nmax || continue
                for beam in 1:cp.numofbeam
                    value = Complex{T}(factor * source[X][input_order + physical_nmax + 1, beam])
                    row = n + nmax + 1
                    if t == 0
                        blocks[1][Y][row, beam] += value
                    else
                        cosine = abs(t) == 2 ? 2 : 4
                        blocks[cosine][Y][row, beam] += value
                        blocks[cosine + 1][Y][row, beam] += (t > 0 ? im : -im) * value
                    end
                end
            end
        end
        if lowell !== nothing && l < 2
            blocks[2][1] .+= lowell[l][1]
            blocks[3][1] .+= lowell[l][2]
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
    return PreparedHWPBlm{T}(components, collect(_HWP_MODES)[active], M, lowell_hwp,
                            cp.lstart, cp.lstop, cp.lmax, cp.bnmax,
                            effective_bnmax, cp.numofbeam)
end

prepare_hwp_blm(cp::ConvolutionParams, blm::AbstractMatrix; kwargs...) =
    prepare_hwp_blm(cp, [blm]; kwargs...)

function beam_batches(blm::PreparedHWPBlm{T}, batch_size::Integer) where {T}
    batches = [beam_batches(component, batch_size) for component in blm.components]
    return [(indices=batches[1][j].indices,
             blm=PreparedHWPBlm{T}([batch[j].blm for batch in batches], copy(blm.modes),
                  copy(blm.mueller), blm.lowell_hwp, blm.lstart, blm.lstop, blm.lmax, blm.bnmax,
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
