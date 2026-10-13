"""
    GPUAlm{T}

Sky harmonic coefficients resident on one CUDA device. `values` has layout
`values[m_index, spin, sky, l_index]`, where spin is ordered `(0,+2,-2)`.
For each `l`, the valid leading part of the first axis is `m=-l:l`; the
remaining padded entries are zero.
"""
struct GPUAlm{T<:_HARMONIC_PRECISIONS}
    values::CuArray{Complex{T},4}
    lstart::Int
    lstop::Int
    lmax::Int
    numofsky::Int
end

Base.eltype(::Type{GPUAlm{T}}) where {T} = Complex{T}
Base.eltype(::GPUAlm{T}) where {T} = Complex{T}

function Base.show(io::IO, alm::GPUAlm{T}) where {T}
    print(
        io,
        "GPUAlm{$T}(l=$(alm.lstart):$(alm.lstop), " *
        "numofsky=$(alm.numofsky), size=$(size(alm.values)))",
    )
end

function _check_prepared_alm_for_upload(alm::PreparedAlm)
    0 <= alm.lstart <= alm.lstop <= alm.lmax || throw(ArgumentError(
        "PreparedAlm bounds must satisfy 0 <= lstart <= lstop <= lmax",
    ))
    alm.numofsky > 0 || throw(ArgumentError(
        "PreparedAlm must contain at least one sky",
    ))
    length(alm.slices) == alm.lstop - alm.lstart + 1 ||
        throw(DimensionMismatch(
            "PreparedAlm slice count does not match its l bounds",
        ))
    for l in alm.lstart:alm.lstop
        blocks = alm_slice(alm, l)
        for spin_index in 1:3
            size(blocks[spin_index]) == (alm.numofsky, 2 * l + 1) ||
                throw(DimensionMismatch(
                    "PreparedAlm spin block has the wrong shape for l=$l",
                ))
        end
    end
    return nothing
end

"""
    upload_alm_gpu(prepared_alm; device_id=nothing) -> GPUAlm

Pack ragged, host-resident `PreparedAlm` slices into one contiguous CUDA
array. The result is independent of the scan ring and can be reused for all
rings and all `l` tiles on the selected device.
"""
function upload_alm_gpu(
    alm::PreparedAlm{T};
    device_id::Union{Nothing,Integer}=nothing,
) where {T<:_HARMONIC_PRECISIONS}
    _check_prepared_alm_for_upload(alm)
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    device_id === nothing || CUDA.device!(Int(device_id))

    l_count = alm.lstop - alm.lstart + 1
    packed = zeros(
        Complex{T},
        2 * alm.lstop + 1,
        3,
        alm.numofsky,
        l_count,
    )
    for (l_index, l) in enumerate(alm.lstart:alm.lstop)
        blocks = alm_slice(alm, l)
        m_count = 2 * l + 1
        for spin_index in 1:3
            @views packed[
                1:m_count,
                spin_index,
                :,
                l_index,
            ] .= permutedims(blocks[spin_index], (2, 1))
        end
    end

    return GPUAlm{T}(
        CuArray(packed),
        alm.lstart,
        alm.lstop,
        alm.lmax,
        alm.numofsky,
    )
end
