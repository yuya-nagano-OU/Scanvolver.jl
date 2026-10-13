const _HARMONIC_PRECISIONS = Union{Float32,Float64}

"""
    PreparedAlm{T}

Harmonic sky coefficients arranged for convolution. `slices[l_index]`
contains the three spin blocks `(0, +2, -2)` for
`l = lstart + l_index - 1`. Each block is a matrix with axes
`(sky, m)` and columns ordered as `m = -l:l`.
"""
struct PreparedAlm{T<:_HARMONIC_PRECISIONS}
    slices::Vector{NTuple{3,Matrix{Complex{T}}}}
    lstart::Int
    lstop::Int
    lmax::Int
    numofsky::Int
end

Base.length(alm::PreparedAlm) = length(alm.slices)
Base.eltype(::Type{PreparedAlm{T}}) where {T} = Complex{T}
Base.eltype(::PreparedAlm{T}) where {T} = Complex{T}

function Base.show(io::IO, alm::PreparedAlm{T}) where {T}
    print(
        io,
        "PreparedAlm{$T}(l=$(alm.lstart):$(alm.lstop), " *
        "numofsky=$(alm.numofsky))",
    )
end

"""
    alm_slice(alm::PreparedAlm, l)

Return the three `(0, +2, -2)` spin blocks for multipole `l`.
Each block has shape `(alm.numofsky, 2l + 1)` and its second axis is
ordered as `m = -l:l`.
"""
function alm_slice(alm::PreparedAlm, l::Integer)
    alm.lstart <= l <= alm.lstop || throw(ArgumentError(
        "l must be in $(alm.lstart):$(alm.lstop), got $l",
    ))
    return alm.slices[Int(l) - alm.lstart + 1]
end

@inline function _packed_alm_index(l::Int, m::Int, lmax::Int)
    return m * (lmax + 1) - (m * (m - 1)) ÷ 2 + (l - m) + 1
end

@inline function _positive_spin_coefficients(
    alm::AbstractMatrix,
    index::Int,
    input_basis::Symbol,
    ::Type{T},
) where {T<:_HARMONIC_PRECISIONS}
    spin0 = convert(Complex{T}, alm[1, index])
    second = convert(Complex{T}, alm[2, index])
    third = convert(Complex{T}, alm[3, index])
    if input_basis === :teb
        spin_plus2 = -(second + Complex{T}(0, 1) * third)
        spin_minus2 = -(second - Complex{T}(0, 1) * third)
        return spin0, spin_plus2, spin_minus2
    end
    return spin0, second, third
end

function _check_alm_inputs(
    cp::ConvolutionParams,
    alms::AbstractVector,
    input_basis::Symbol,
    precision::Type,
)
    input_basis in (:teb, :spin) || throw(ArgumentError(
        "input_basis must be :teb or :spin, got $input_basis",
    ))
    precision in (Float32, Float64) || throw(ArgumentError(
        "precision must be Float32 or Float64, got $precision",
    ))
    isempty(alms) && throw(ArgumentError(
        "alms must contain at least one sky",
    ))
    length(alms) == cp.numofsky || throw(ArgumentError(
        "expected cp.numofsky=$(cp.numofsky) sky inputs, got $(length(alms))",
    ))

    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    for (sky_index, alm) in pairs(alms)
        alm isa AbstractMatrix || throw(ArgumentError(
            "alms[$sky_index] must be a matrix with axes (T/E/B or spin, coefficient)",
        ))
        size(alm) == (3, coefficient_count) || throw(DimensionMismatch(
            "alms[$sky_index] has size $(size(alm)); expected " *
            "(3, $coefficient_count) for cp.lmax=$(cp.lmax)",
        ))
    end
    return nothing
end

"""
    prepare_alm(cp, alm; input_basis=:teb, precision=Float64)
    prepare_alm(cp, alms; input_basis=:teb, precision=Float64)

Prepare one or more skies for convolution from HEALPix/healpy packed
non-negative-`m` coefficients. Each input sky is a matrix with shape
`(3, (lmax+1)(lmax+2)/2)`.

With `input_basis=:teb`, the three rows are interpreted as `(T,E,B)` and
converted to `(spin 0, spin +2, spin -2)` using

```math
a_{+2}=-(a_E+i a_B), \\qquad a_{-2}=-(a_E-i a_B).
```

With `input_basis=:spin`, the rows are already `(0,+2,-2)`. In both cases,
negative-`m` coefficients are reconstructed using
`a_s(l,-m)=(-1)^m conj(a_{-s}(l,m))`.

The result stores one ragged slice per `l`, avoiding padding every multipole
to `2*lstop+1` before the later GPU upload step.
"""
function prepare_alm(
    cp::ConvolutionParams,
    alm::AbstractMatrix;
    input_basis::Symbol=:teb,
    precision::Type=Float64,
)
    return prepare_alm(
        cp,
        [alm];
        input_basis=input_basis,
        precision=precision,
    )
end

function prepare_alm(
    cp::ConvolutionParams,
    alms::AbstractVector;
    input_basis::Symbol=:teb,
    precision::Type=Float64,
)
    _check_alm_inputs(cp, alms, input_basis, precision)
    return _prepare_alm(cp, alms, input_basis, precision)
end

function _prepare_alm(
    cp::ConvolutionParams,
    alms::AbstractVector,
    input_basis::Symbol,
    ::Type{T},
) where {T<:_HARMONIC_PRECISIONS}
    slices = Vector{NTuple{3,Matrix{Complex{T}}}}(
        undef,
        cp.lstop - cp.lstart + 1,
    )

    for (l_index, l) in enumerate(cp.lstart:cp.lstop)
        blocks = ntuple(
            _ -> zeros(Complex{T}, cp.numofsky, 2 * l + 1),
            3,
        )
        for m in 0:l
            packed_index = _packed_alm_index(l, m, cp.lmax)
            positive_index = m + l + 1
            negative_index = -m + l + 1
            phase = isodd(m) ? -one(T) : one(T)
            for sky_index in eachindex(alms)
                spin0, spin_plus2, spin_minus2 =
                    _positive_spin_coefficients(
                        alms[sky_index],
                        packed_index,
                        input_basis,
                        T,
                    )
                blocks[1][sky_index, positive_index] = spin0
                blocks[2][sky_index, positive_index] = spin_plus2
                blocks[3][sky_index, positive_index] = spin_minus2
                if m > 0
                    blocks[1][sky_index, negative_index] =
                        phase * conj(spin0)
                    blocks[2][sky_index, negative_index] =
                        phase * conj(spin_minus2)
                    blocks[3][sky_index, negative_index] =
                        phase * conj(spin_plus2)
                end
            end
        end
        slices[l_index] = blocks
    end

    return PreparedAlm{T}(
        slices,
        cp.lstart,
        cp.lstop,
        cp.lmax,
        cp.numofsky,
    )
end
