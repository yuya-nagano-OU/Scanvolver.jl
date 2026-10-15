"""
    PreparedBlm{T}

Beam harmonic coefficients arranged for convolution. `slices[l_index]`
contains the three spin blocks `(0, +2, -2)` for
`l = lstart + l_index - 1`. Each block is a matrix with axes `(n, beam)`;
its rows are ordered as `n = -nmax(l):nmax(l)`.

The stored values include the complex conjugation required by the
convolution contraction.
"""
struct PreparedBlm{T<:_HARMONIC_PRECISIONS}
    slices::Vector{NTuple{3,Matrix{Complex{T}}}}
    lstart::Int
    lstop::Int
    lmax::Int
    bnmax::Int
    numofbeam::Int
end

Base.length(blm::PreparedBlm) = length(blm.slices)
Base.eltype(::Type{PreparedBlm{T}}) where {T} = Complex{T}
Base.eltype(::PreparedBlm{T}) where {T} = Complex{T}

function Base.show(io::IO, blm::PreparedBlm{T}) where {T}
    print(
        io,
        "PreparedBlm{$T}(l=$(blm.lstart):$(blm.lstop), " *
        "bnmax=$(blm.bnmax), numofbeam=$(blm.numofbeam))",
    )
end

"""
    blm_slice(blm::PreparedBlm, l)

Return the three `(0, +2, -2)` spin blocks for multipole `l`.
Each block has shape `(2*nmax(l)+1, blm.numofbeam)`, with its first axis
ordered from negative to positive `n`.
"""
function blm_slice(blm::PreparedBlm, l::Integer)
    blm.lstart <= l <= blm.lstop || throw(ArgumentError(
        "l must be in $(blm.lstart):$(blm.lstop), got $l",
    ))
    return blm.slices[Int(l) - blm.lstart + 1]
end

function _check_blm_inputs(
    cp::ConvolutionParams,
    blms::AbstractVector,
    input_basis::Symbol,
    precision::Type,
)
    cp.HWP == false || throw(ArgumentError(
        "prepare_blm currently supports HWP=false",
    ))
    input_basis in (:teb, :spin) || throw(ArgumentError(
        "input_basis must be :teb or :spin, got $input_basis",
    ))
    precision in (Float32, Float64) || throw(ArgumentError(
        "precision must be Float32 or Float64, got $precision",
    ))
    isempty(blms) && throw(ArgumentError(
        "blms must contain at least one beam",
    ))
    length(blms) == cp.numofbeam || throw(ArgumentError(
        "expected cp.numofbeam=$(cp.numofbeam) beam inputs, got $(length(blms))",
    ))

    coefficient_count = (cp.lmax + 1) * (cp.lmax + 2) ÷ 2
    for (beam_index, blm) in pairs(blms)
        blm isa AbstractMatrix || throw(ArgumentError(
            "blms[$beam_index] must be a matrix with axes (T/E/B or spin, coefficient)",
        ))
        size(blm) == (3, coefficient_count) || throw(DimensionMismatch(
            "blms[$beam_index] has size $(size(blm)); expected " *
            "(3, $coefficient_count) for cp.lmax=$(cp.lmax)",
        ))
    end
    return nothing
end

"""
    prepare_blm(cp, blm; input_basis=:teb, precision=Float64)
    prepare_blm(cp, blms; input_basis=:teb, precision=Float64)

Prepare one or more beams for convolution from HEALPix/healpy packed
non-negative-`n` coefficients. Each input beam is a matrix with shape
`(3, (lmax+1)(lmax+2)/2)`.

With `input_basis=:teb`, the rows are converted to `(0,+2,-2)` using

```math
b_{+2}=-(b_E+i b_B), \\qquad b_{-2}=-(b_E-i b_B).
```

With `input_basis=:spin`, the rows are already `(0,+2,-2)`. The prepared
coefficient stored at positive `n` is the complex conjugate of the input
spin coefficient. Negative modes obey

```math
B_s(l,-n)=(-1)^n B_{-s}(l,n)^*,
```

where `B` denotes the prepared, conjugated coefficient. The three returned
blocks are always ordered `(0,+2,-2)`.

Only modes `-calc_nmax(cp,l):calc_nmax(cp,l)` are retained for each `l`.
The current GPU convolution path supports `HWP=false` only, and this
preparation function rejects HWP-enabled parameters accordingly.
"""
function prepare_blm(
    cp::ConvolutionParams,
    blm::AbstractMatrix;
    input_basis::Symbol=:teb,
    precision::Type=Float64,
)
    return prepare_blm(
        cp,
        [blm];
        input_basis=input_basis,
        precision=precision,
    )
end

function prepare_blm(
    cp::ConvolutionParams,
    blms::AbstractVector;
    input_basis::Symbol=:teb,
    precision::Type=Float64,
)
    _check_blm_inputs(cp, blms, input_basis, precision)
    return _prepare_blm(cp, blms, input_basis, precision)
end

function _prepare_blm(
    cp::ConvolutionParams,
    blms::AbstractVector,
    input_basis::Symbol,
    ::Type{T},
) where {T<:_HARMONIC_PRECISIONS}
    slices = Vector{NTuple{3,Matrix{Complex{T}}}}(
        undef,
        cp.lstop - cp.lstart + 1,
    )

    for (l_index, l) in enumerate(cp.lstart:cp.lstop)
        nmax = calc_nmax(cp, l)
        blocks = ntuple(
            _ -> zeros(Complex{T}, 2 * nmax + 1, cp.numofbeam),
            3,
        )
        for n in 0:nmax
            packed_index = _packed_alm_index(l, n, cp.lmax)
            positive_index = n + nmax + 1
            negative_index = -n + nmax + 1
            phase = isodd(n) ? -one(T) : one(T)
            for beam_index in eachindex(blms)
                spin0, spin_plus2, spin_minus2 =
                    _positive_spin_coefficients(
                        blms[beam_index],
                        packed_index,
                        input_basis,
                        T,
                    )

                # Beam coefficients enter the convolution conjugated.
                blocks[1][positive_index, beam_index] = conj(spin0)
                blocks[2][positive_index, beam_index] = conj(spin_plus2)
                blocks[3][positive_index, beam_index] = conj(spin_minus2)
                if n > 0
                    # This is conjugate(±1 reconstructed input).  The spin
                    # +2/-2 blocks swap under n -> -n.
                    blocks[1][negative_index, beam_index] = phase * spin0
                    blocks[2][negative_index, beam_index] =
                        phase * spin_minus2
                    blocks[3][negative_index, beam_index] =
                        phase * spin_plus2
                end
            end
        end
        slices[l_index] = blocks
    end

    return PreparedBlm{T}(
        slices,
        cp.lstart,
        cp.lstop,
        cp.lmax,
        cp.bnmax,
        cp.numofbeam,
    )
end

"""Split already-prepared, conjugated beams without conjugating them again."""
function beam_batches(blm::PreparedBlm{T}, batch_size::Integer) where {T}
    batch_size > 0 || throw(ArgumentError("beam_batch_size must be positive"))
    return [
        (indices=first:min(first + Int(batch_size) - 1, blm.numofbeam),
         blm=PreparedBlm{T}(
             [ntuple(spin -> block[spin][:, first:min(first + Int(batch_size) - 1, blm.numofbeam)], 3)
              for block in blm.slices],
             blm.lstart, blm.lstop, blm.lmax, blm.bnmax,
             min(Int(batch_size), blm.numofbeam - first + 1),
         ))
        for first in 1:Int(batch_size):blm.numofbeam
    ]
end
