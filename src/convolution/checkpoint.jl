# Keep the pre-rename marker so checkpoints written by EBCoDeGPU remain
# readable after the package becomes EBCoDeCMB.
const _CHECKPOINT_MAGIC = b"EBCoDeGPU-ckpt\x00\x00"
const _CHECKPOINT_VERSION = 2
const _CHECKPOINT_SIGNATURE_BYTES = 32
const _CheckpointSignature = NTuple{_CHECKPOINT_SIGNATURE_BYTES,UInt8}
const _EMPTY_CHECKPOINT_SIGNATURE = ntuple(
    _ -> UInt8(0),
    _CHECKPOINT_SIGNATURE_BYTES,
)

"""
    SphereCheckpoint{T}

A resumable state of `convolve_sphere_gpu` in `:full_map` mode: which rings of
`first_ring:last_ring` are finished, and the partial map.

Only `:full_map` is checkpointed, because it is the only mode where the driver
owns the accumulated result. Under `:ring_callback` or `:discard` the output
belongs to the caller, whose own sink would have to be resumable too, and
recording ring completion without the data it produced would invite resuming
onto an inconsistent sink.
"""
struct SphereCheckpoint{T<:_HARMONIC_PRECISIONS}
    first_ring::Int
    last_ring::Int
    completed::Vector{Bool}
    stokes_map::Array{T,4}
    signature::_CheckpointSignature
end

function SphereCheckpoint(
    first_ring::Int,
    last_ring::Int,
    completed::Vector{Bool},
    stokes_map::Array{T,4},
) where {T<:_HARMONIC_PRECISIONS}
    return SphereCheckpoint{T}(
        first_ring,
        last_ring,
        completed,
        stokes_map,
        _EMPTY_CHECKPOINT_SIGNATURE,
    )
end

function Base.show(io::IO, checkpoint::SphereCheckpoint{T}) where {T}
    print(
        io,
        "SphereCheckpoint{$T}(rings=$(checkpoint.first_ring):",
        "$(checkpoint.last_ring), ",
        "$(count(checkpoint.completed))/$(length(checkpoint.completed)) done)",
    )
end

_checkpoint_precision_tag(::Type{Float32}) = 4
_checkpoint_precision_tag(::Type{Float64}) = 8

"""
    _sphere_checkpoint_signature(checkpoint_id, cp, buffer, alm, blm,
        table, scan_plan, rings; ...)

Build the deterministic signature used by the sphere drivers. The explicit
`checkpoint_id` is the caller's provenance key for the potentially very large
alm, blm, and pointing contents; the remaining fields prevent the same key
from being reused accidentally with different calculation settings or input
shapes without reading those arrays back from the GPU or hashing a multi-GB
scan buffer.
"""
function _sphere_checkpoint_signature(
    checkpoint_id::AbstractString,
    cp::ConvolutionParams,
    buffer::RingScanBuffer,
    alm,
    blm::PreparedBlm,
    table,
    scan_plan::ScanMomentPlan,
    rings::AbstractUnitRange{<:Integer};
    compute_precision::Type,
    map_precision::Type,
    rcond::Real,
    unseen::Real,
    moment_method::Symbol,
    l_tile,
    memory_fraction::Real,
)
    descriptor = (
        signature_version=1,
        checkpoint_id=String(checkpoint_id),
        convolution=(
            nside=cp.nside_output,
            lstart=cp.lstart,
            lstop=cp.lstop,
            lmax=cp.lmax,
            numofsky=cp.numofsky,
            numofbeam=cp.numofbeam,
            bnmax=cp.bnmax,
            kappa=cp.κ,
            hwp=cp.HWP,
            hwp_rpm=cp.HWP_rpm,
            noise=cp.noise,
            noise_sigma=cp.noise_sigma,
        ),
        rings=(first=Int(first(rings)), last=Int(last(rings))),
        buffer=(
            nside=buffer.nside,
            first_ring=buffer.first_ring,
            last_ring=buffer.last_ring,
            first_pixel=buffer.first_pixel,
            last_pixel=buffer.last_pixel,
            processed_samples=buffer.processed_samples,
            stored_samples=buffer.stored_samples,
            offset_count=length(buffer.pixel_offsets),
        ),
        alm=(
            lstart=alm.lstart,
            lstop=alm.lstop,
            lmax=alm.lmax,
            numofsky=alm.numofsky,
            precision=string(eltype(alm)),
        ),
        blm=(
            lstart=blm.lstart,
            lstop=blm.lstop,
            lmax=blm.lmax,
            bnmax=blm.bnmax,
            numofbeam=blm.numofbeam,
            precision=string(eltype(blm)),
        ),
        wigner_table=(
            lstart=Int(table.lstart),
            lstop=Int(table.lstop),
            kappa=Int(table.kappa),
            nmax=Int(table.nmax),
            beta_max=repr(table.beta_max),
            coefficient_size=size(table.coefficients),
            precision=string(eltype(table.coefficients)),
        ),
        scan_plan=(
            kappa=scan_plan.kappa,
            nmax=scan_plan.nmax,
            beta_order=scan_plan.beta_order,
            beta_max=repr(scan_plan.beta_max),
            precision=string(typeof(scan_plan.beta_max)),
        ),
        calculation=(
            compute_precision=string(compute_precision),
            map_precision=string(map_precision),
            rcond=(type=string(typeof(rcond)), value=repr(rcond)),
            unseen=(type=string(typeof(unseen)), value=repr(unseen)),
            moment_method=moment_method,
            l_tile=l_tile,
            memory_fraction=(
                type=string(typeof(memory_fraction)),
                value=repr(memory_fraction),
            ),
        ),
    )
    digest = sha256(codeunits(repr(descriptor)))
    return ntuple(index -> digest[index], _CHECKPOINT_SIGNATURE_BYTES)
end

function _checkpoint_header(cp::ConvolutionParams, checkpoint::SphereCheckpoint{T}) where {T}
    return Int64[
        _CHECKPOINT_VERSION,
        cp.nside_output,
        cp.numofsky,
        cp.numofbeam,
        cp.lstart,
        cp.lstop,
        cp.lmax,
        checkpoint.first_ring,
        checkpoint.last_ring,
        _checkpoint_precision_tag(T),
        cp.resol.numOfPixels,
    ]
end

"""
    save_sphere_checkpoint(path, cp, checkpoint)

Write a checkpoint, replacing any previous one at `path` atomically: the bytes
go to `path * ".tmp"` and are renamed only once complete, so an interrupted
write cannot leave a half-file that a later run would trust.

The format is a fixed `Int64` header, a SHA-256 run signature, one byte per
ring, then the raw map.
"""
function save_sphere_checkpoint(
    path::AbstractString,
    cp::ConvolutionParams,
    checkpoint::SphereCheckpoint{T},
) where {T<:_HARMONIC_PRECISIONS}
    size(checkpoint.stokes_map) == (
        cp.numofsky,
        cp.numofbeam,
        3,
        cp.resol.numOfPixels,
    ) || throw(DimensionMismatch(
        "stokes_map has size $(size(checkpoint.stokes_map)); expected " *
        "($(cp.numofsky), $(cp.numofbeam), 3, $(cp.resol.numOfPixels))",
    ))
    length(checkpoint.completed) ==
        checkpoint.last_ring - checkpoint.first_ring + 1 ||
        throw(DimensionMismatch(
            "completed has $(length(checkpoint.completed)) entries for " *
            "rings $(checkpoint.first_ring):$(checkpoint.last_ring)",
        ))

    temporary = path * ".tmp"
    open(temporary, "w") do io
        write(io, _CHECKPOINT_MAGIC)
        write(io, _checkpoint_header(cp, checkpoint))
        write(io, collect(checkpoint.signature))
        write(io, UInt8[flag ? 0x01 : 0x00 for flag in checkpoint.completed])
        write(io, checkpoint.stokes_map)
    end
    mv(temporary, path; force=true)
    return path
end

"""
    load_sphere_checkpoint(path, cp; expected_signature=nothing)
        -> SphereCheckpoint

Read a checkpoint and verify that it describes the same problem as `cp`. A
mismatch in resolution, sky or beam count, or multipole range throws rather
than resuming onto an incompatible run.
"""
function load_sphere_checkpoint(
    path::AbstractString,
    cp::ConvolutionParams,
    ;
    expected_signature::Union{Nothing,_CheckpointSignature}=nothing,
)
    open(path, "r") do io
        magic = read(io, length(_CHECKPOINT_MAGIC))
        magic == _CHECKPOINT_MAGIC || throw(ArgumentError(
            "$path is not a compatible EBCoDeCMB checkpoint",
        ))
        header = Vector{Int64}(undef, 11)
        read!(io, header)
        header[1] == _CHECKPOINT_VERSION || throw(ArgumentError(
            "$path has checkpoint version $(header[1]); this build reads " *
            "version $_CHECKPOINT_VERSION",
        ))
        expected = Int64[
            cp.nside_output,
            cp.numofsky,
            cp.numofbeam,
            cp.lstart,
            cp.lstop,
            cp.lmax,
        ]
        header[2:7] == expected || throw(ArgumentError(
            "$path was written for (nside, numofsky, numofbeam, lstart, " *
            "lstop, lmax) = $(header[2:7]); this run has $expected",
        ))
        first_ring = Int(header[8])
        last_ring = Int(header[9])
        precision = header[10] == 4 ? Float32 :
            header[10] == 8 ? Float64 :
            throw(ArgumentError(
                "$path has an unknown precision tag $(header[10])",
            ))
        pixel_count = Int(header[11])
        pixel_count == cp.resol.numOfPixels || throw(ArgumentError(
            "$path holds $pixel_count pixels; this run has " *
            "$(cp.resol.numOfPixels)",
        ))

        signature_bytes = Vector{UInt8}(undef, _CHECKPOINT_SIGNATURE_BYTES)
        read!(io, signature_bytes)
        signature = ntuple(
            index -> signature_bytes[index],
            _CHECKPOINT_SIGNATURE_BYTES,
        )
        if expected_signature !== nothing && signature != expected_signature
            throw(ArgumentError(
                "$path belongs to a different checkpoint_id, input set, " *
                "or calculation configuration",
            ))
        end

        ring_count = last_ring - first_ring + 1
        1 <= first_ring <= last_ring <= 4 * cp.nside_output - 1 ||
            throw(ArgumentError(
                "$path has invalid ring bounds $first_ring:$last_ring",
            ))
        flags = Vector{UInt8}(undef, ring_count)
        read!(io, flags)
        all(flag -> flag == 0x00 || flag == 0x01, flags) ||
            throw(ArgumentError("$path has an invalid completion flag"))
        stokes_map = Array{precision,4}(
            undef,
            cp.numofsky,
            cp.numofbeam,
            3,
            pixel_count,
        )
        read!(io, stokes_map)
        eof(io) || throw(ArgumentError(
            "$path has unexpected trailing data",
        ))
        return SphereCheckpoint{precision}(
            first_ring,
            last_ring,
            [flag != 0x00 for flag in flags],
            stokes_map,
            signature,
        )
    end
end
