#!/usr/bin/env julia

using Pkg

function find_repo_root(start::AbstractString=@__DIR__)
    current = abspath(start)
    while true
        if isfile(joinpath(current, "Project.toml")) &&
            isfile(joinpath(current, "src", "EBCoDeCMB.jl"))
            return current
        end
        parent = dirname(current)
        parent == current && error(
            "EBCoDeCMB.jl repository root was not found",
        )
        current = parent
    end
end

const REPO_ROOT = find_repo_root()
const SCRIPT_DIR = @__DIR__
Pkg.activate(SCRIPT_DIR)
if !isfile(joinpath(SCRIPT_DIR, "Manifest.toml"))
    println("Instantiating script environment at $SCRIPT_DIR")
    Pkg.instantiate()
end

using CUDA
using Dates
using EBCoDeCMB
using Falcons
using Healpix: pix2angRing
using NPZ: npzread, npzwrite
using Printf
using SHA
using Statistics

# -----------------------------------------------------------------------------
# User-editable configuration
#
# Every setting can also be overridden with EBCODE_<NAME>.  For example:
# EBCODE_NSIDE=32 EBCODE_DURATION_SECONDS=3600 \
#   julia --project=script/A6000_double script/A6000_double/run_convolution.jl
# Devices are a comma-separated list, for example EBCODE_DEVICES=0,1.
# -----------------------------------------------------------------------------

ALM_PATH = joinpath(REPO_ROOT, "test", "data", "alm_cmb_r0_nside16.npy")
BLM_PATH = joinpath(REPO_ROOT, "test", "data", "blm_gauss_fwhm5_nside16.npy")
OUTPUT_DIR = joinpath(SCRIPT_DIR, "output")
RUN_NAME = "nside16_lmax47_1day"

NSIDE = 16
LSTART = 0
LSTOP = 47
LMAX = 47
BNMAX = 5
KAPPA = 10

SAMPLING_RATE = 1.0
DURATION_SECONDS = 24 * 60 * 60
CHUNK_SECONDS = 6 * 60 * 60

PRECISION = Float64
MAP_PRECISION = Float64
BETA_ORDER = 16
BETA_MARGIN = 1.02
RCOND = 1e-2
MOMENT_METHOD = :gemm
L_TILE = :auto
MEMORY_FRACTION = 0.8

DEVICES = [0, 1]
RECLAIM_INTERVAL = 16
CHECKPOINT_INTERVAL = 16
RESUME = true
OUTPUT_MODE = :full_map

const MEMORY_ESTIMATE_FORMAT = Printf.Format(
    "Memory estimate (ring %d, %d/%d active): persistent %.2f MiB, " *
    "tile-1 %.2f MiB, total %.2f MiB\n",
)
const CONVOLUTION_PROGRESS_FORMAT = Printf.Format(
    "convolution: %d/%d rings, latest ring %d on GPU %d, " *
    "%d active pixels, %d hits, %.1f s\n",
)
const POINTING_BUFFER_FORMAT = Printf.Format(
    "Pointing buffer: %d retained / %d generated samples, %.2f MiB, " *
    "%.1f s\n",
)
const WIGNER_TABLE_FORMAT = Printf.Format(
    "Built Wigner-Chebyshev table in %.1f s: %d pairs, %d l values, " *
    "order %d\n",
)

"""
    setting(name, default, parser=identity)

Return `default` unless `EBCODE_<name>` is present, in which case parse the
environment value with `parser`.  This is the sole environment override path.
"""
function setting(name::AbstractString, default, parser=identity)
    value = get(ENV, "EBCODE_" * name, nothing)
    return value === nothing ? default : parser(value)
end

function parse_bool(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized in ("1", "true", "yes", "on") && return true
    normalized in ("0", "false", "no", "off") && return false
    throw(ArgumentError(
        "expected a Boolean value (true/false or 1/0), got $value",
    ))
end

function parse_precision(value::AbstractString)
    normalized = strip(value)
    normalized == "Float32" && return Float32
    normalized == "Float64" && return Float64
    throw(ArgumentError(
        "precision must be Float32 or Float64, got $value",
    ))
end

function parse_devices(value::AbstractString)
    fields = split(strip(value), ',')
    isempty(fields) && throw(ArgumentError("devices must not be empty"))
    devices = Int[]
    for field in fields
        isempty(strip(field)) && throw(ArgumentError(
            "devices must be a comma-separated list of integers, got $value",
        ))
        push!(devices, parse(Int, strip(field)))
    end
    return devices
end

function parse_output_mode(value::AbstractString)
    mode = Symbol(strip(value))
    mode in (:full_map, :ring_callback, :discard) || throw(ArgumentError(
        "OUTPUT_MODE must be :full_map, :ring_callback, or :discard; " *
        "got $value",
    ))
    return mode
end

function parse_moment_method(value::AbstractString)
    method = Symbol(strip(value))
    method in (:gemm, :direct) || throw(ArgumentError(
        "MOMENT_METHOD must be :gemm or :direct, got $value",
    ))
    return method
end

function parse_l_tile(value::AbstractString)
    normalized = strip(value)
    normalized == "auto" && return :auto
    tile = parse(Int, normalized)
    tile > 0 || throw(ArgumentError("L_TILE must be positive, got $tile"))
    return tile
end

function resolved_settings()
    return (
        alm_path=abspath(expanduser(setting("ALM_PATH", ALM_PATH, String))),
        blm_path=abspath(expanduser(setting("BLM_PATH", BLM_PATH, String))),
        output_dir=abspath(expanduser(
            setting("OUTPUT_DIR", OUTPUT_DIR, String),
        )),
        run_name=setting("RUN_NAME", RUN_NAME, String),
        nside=setting("NSIDE", NSIDE, value -> parse(Int, value)),
        lstart=setting("LSTART", LSTART, value -> parse(Int, value)),
        lstop=setting("LSTOP", LSTOP, value -> parse(Int, value)),
        lmax=setting("LMAX", LMAX, value -> parse(Int, value)),
        bnmax=setting("BNMAX", BNMAX, value -> parse(Int, value)),
        kappa=setting("KAPPA", KAPPA, value -> parse(Int, value)),
        sampling_rate=setting(
            "SAMPLING_RATE", SAMPLING_RATE, value -> parse(Float64, value),
        ),
        duration_seconds=setting(
            "DURATION_SECONDS", DURATION_SECONDS,
            value -> parse(Float64, value),
        ),
        chunk_seconds=setting(
            "CHUNK_SECONDS", CHUNK_SECONDS, value -> parse(Float64, value),
        ),
        precision=setting("PRECISION", PRECISION, parse_precision),
        map_precision=setting(
            "MAP_PRECISION", MAP_PRECISION, parse_precision,
        ),
        beta_order=setting(
            "BETA_ORDER", BETA_ORDER, value -> parse(Int, value),
        ),
        beta_margin=setting(
            "BETA_MARGIN", BETA_MARGIN, value -> parse(Float64, value),
        ),
        rcond=setting("RCOND", RCOND, value -> parse(Float64, value)),
        moment_method=setting(
            "MOMENT_METHOD", MOMENT_METHOD, parse_moment_method,
        ),
        l_tile=setting("L_TILE", L_TILE, parse_l_tile),
        memory_fraction=setting(
            "MEMORY_FRACTION", MEMORY_FRACTION, value -> parse(Float64, value),
        ),
        devices=setting("DEVICES", DEVICES, parse_devices),
        reclaim_interval=setting(
            "RECLAIM_INTERVAL", RECLAIM_INTERVAL, value -> parse(Int, value),
        ),
        checkpoint_interval=setting(
            "CHECKPOINT_INTERVAL", CHECKPOINT_INTERVAL,
            value -> parse(Int, value),
        ),
        resume=setting("RESUME", RESUME, parse_bool),
        output_mode=setting("OUTPUT_MODE", OUTPUT_MODE, parse_output_mode),
    )
end

function print_settings(settings)
    println("Resolved configuration:")
    for (name, value) in pairs(settings)
        @printf("  %-22s = %s\n", string(name), repr(value))
    end
end

function reject_unsupported_options()
    for name in ("HWP", "NOISE", "HWP_RPM", "NOISE_SIGMA")
        haskey(ENV, "EBCODE_" * name) || continue
        throw(ArgumentError(
            "EBCODE_$name was specified, but this script supports only " *
            "HWP=false and noise=false",
        ))
    end
end

function validate_settings(settings)
    isempty(strip(settings.run_name)) && throw(ArgumentError(
        "RUN_NAME must not be empty",
    ))
    isfinite(settings.sampling_rate) && settings.sampling_rate > 0 ||
        throw(ArgumentError(
            "SAMPLING_RATE must be finite and positive, got " *
            "$(settings.sampling_rate)",
        ))
    isfinite(settings.duration_seconds) && settings.duration_seconds > 0 ||
        throw(ArgumentError(
            "DURATION_SECONDS must be finite and positive, got " *
            "$(settings.duration_seconds)",
        ))
    isfinite(settings.chunk_seconds) && settings.chunk_seconds > 0 ||
        throw(ArgumentError(
            "CHUNK_SECONDS must be finite and positive, got " *
            "$(settings.chunk_seconds)",
        ))
    settings.beta_order > 0 || throw(ArgumentError(
        "BETA_ORDER must be positive, got $(settings.beta_order)",
    ))
    isfinite(settings.beta_margin) && settings.beta_margin >= 1 ||
        throw(ArgumentError(
            "BETA_MARGIN must be finite and at least 1, got " *
            "$(settings.beta_margin)",
        ))
    isfinite(settings.rcond) && settings.rcond > 0 || throw(ArgumentError(
        "RCOND must be finite and positive, got $(settings.rcond)",
    ))
    isfinite(settings.memory_fraction) &&
        0 < settings.memory_fraction <= 1 || throw(ArgumentError(
        "MEMORY_FRACTION must be in (0, 1], got " *
        "$(settings.memory_fraction)",
    ))
    settings.reclaim_interval >= 0 || throw(ArgumentError(
        "RECLAIM_INTERVAL must be non-negative, got " *
        "$(settings.reclaim_interval)",
    ))
    settings.checkpoint_interval >= 0 || throw(ArgumentError(
        "CHECKPOINT_INTERVAL must be non-negative, got " *
        "$(settings.checkpoint_interval)",
    ))
    return nothing
end

function packed_lmax(coefficient_count::Integer, label::AbstractString)
    count = Int(coefficient_count)
    count > 0 || throw(ArgumentError(
        "$label has $count coefficients; expected a positive triangular " *
        "HEALPix packed coefficient count",
    ))
    discriminant = 1 + 8 * count
    root = isqrt(discriminant)
    root * root == discriminant && iseven(root - 3) || throw(ArgumentError(
        "$label has $count coefficients; this is not " *
        "(lmax + 1)(lmax + 2) / 2 for an integer lmax",
    ))
    lmax = (root - 3) ÷ 2
    lmax >= 0 || throw(ArgumentError(
        "$label has $count coefficients; inferred invalid lmax=$lmax",
    ))
    return lmax
end

function check_input_path(path::AbstractString, label::AbstractString)
    isfile(path) || throw(ArgumentError(
        "$label file does not exist: $path",
    ))
    lowercase(splitext(path)[2]) == ".npy" || throw(ArgumentError(
        "$label must be a .npy file for NPZ.jl input, got $path",
    ))
    return nothing
end

function load_harmonics(path::AbstractString, label::AbstractString, lmax::Int)
    check_input_path(path, label)
    values = npzread(path)
    values isa AbstractMatrix || throw(ArgumentError(
        "$label has type $(typeof(values)); expected a two-dimensional " *
        "(3, ncoeff) .npy array",
    ))
    size(values, 1) == 3 || throw(DimensionMismatch(
        "$label has size $(size(values)); expected 3 rows ordered as " *
        "(T, E, B)",
    ))
    input_lmax = packed_lmax(size(values, 2), label)
    input_lmax == lmax || throw(ArgumentError(
        "$label has lmax=$input_lmax from $(size(values, 2)) coefficients, " *
        "but LMAX=$lmax. Packed HEALPix coefficients cannot be truncated " *
        "by taking an array prefix; use inputs packed at the requested lmax.",
    ))
    expected = (lmax + 1) * (lmax + 2) ÷ 2
    size(values) == (3, expected) || throw(DimensionMismatch(
        "$label has size $(size(values)); expected (3, $expected) for " *
        "LMAX=$lmax",
    ))
    return ComplexF64.(values)
end

function preflight!(settings)
    reject_unsupported_options()
    validate_settings(settings)
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    available = length(CUDA.devices())
    isempty(settings.devices) && throw(ArgumentError(
        "DEVICES must not be empty",
    ))
    all(device -> 0 <= device < available, settings.devices) ||
        throw(ArgumentError(
            "DEVICES must be in 0:$(available - 1), got $(settings.devices)",
        ))
    if Threads.nthreads() < length(settings.devices)
        thread_hint = "start Julia with JULIA_NUM_THREADS=" *
            string(length(settings.devices))
        @warn(
            "fewer Julia threads than devices; workers will interleave",
            threads=Threads.nthreads(),
            devices=length(settings.devices),
            hint=thread_hint,
        )
    end
    alm = load_harmonics(settings.alm_path, "ALM", settings.lmax)
    blm = load_harmonics(settings.blm_path, "BLM", settings.lmax)
    return alm, blm
end

function pointing_chunks(settings)
    total_samples = round(
        Int,
        settings.duration_seconds * settings.sampling_rate,
    )
    total_samples > 0 || throw(ArgumentError(
        "DURATION_SECONDS * SAMPLING_RATE rounds to $total_samples samples",
    ))
    chunk_samples = round(Int, settings.chunk_seconds * settings.sampling_rate)
    chunk_samples > 0 || throw(ArgumentError(
        "CHUNK_SECONDS * SAMPLING_RATE rounds to $chunk_samples samples",
    ))

    strategy = gen_ScanningStrategy()
    strategy.nside = settings.nside
    strategy.sampling_rate = settings.sampling_rate
    strategy.duration = settings.duration_seconds

    chunks = Channel{NTuple{3,Vector{Float64}}}(2)
    producer = @async begin
        try
            for first_sample in 0:chunk_samples:(total_samples - 1)
                last_sample = min(total_samples, first_sample + chunk_samples)
                theta, phi, psi, _ = get_pointings(
                    strategy,
                    first_sample,
                    last_sample,
                )
                put!(chunks, (
                    vec(Float64.(phi[:, 1])),
                    vec(Float64.(theta[:, 1])),
                    vec(Float64.(psi[:, 1])),
                ))
            end
        finally
            close(chunks)
        end
    end
    return chunks, producer, total_samples
end

function buffer_progress(total_samples::Int)
    next_report = Ref(max(1, cld(total_samples, 10)))
    return record -> begin
        record.processed_samples < next_report[] && return nothing
        @printf(
            "pointing: %d/%d samples, %d retained\n",
            record.processed_samples,
            total_samples,
            record.stored_samples,
        )
        next_report[] += max(1, cld(total_samples, 10))
        return nothing
    end
end

function maximum_buffer_beta(cp::ConvolutionParams, buffer::RingScanBuffer)
    maximum_beta = 0.0
    @inbounds for pixel in buffer.first_pixel:buffer.last_pixel
        bucket = pixel - buffer.first_pixel + 1
        first_sample = buffer.pixel_offsets[bucket]
        last_sample = buffer.pixel_offsets[bucket + 1] - 1
        last_sample >= first_sample || continue

        theta_pixel, phi_pixel = pix2angRing(cp.resol, pixel)
        sin_theta_pixel = sin(theta_pixel)
        cos_theta_pixel = cos(theta_pixel)
        for sample in first_sample:last_sample
            cosine = cos_theta_pixel * cos(buffer.theta[sample]) +
                sin_theta_pixel * sin(buffer.theta[sample]) *
                cos(phi_pixel - buffer.phi[sample])
            maximum_beta = max(maximum_beta, acos(clamp(cosine, -1.0, 1.0)))
        end
    end
    return maximum_beta
end

function largest_buffer_ring(cp::ConvolutionParams, buffer::RingScanBuffer)
    largest = (ring=0, pixels=0, active_pixels=0)
    for ring in buffer.first_ring:buffer.last_ring
        first_pixel, last_pixel = ring_pixel_range(cp, ring)
        active_pixels = 0
        @inbounds for pixel in first_pixel:last_pixel
            bucket = pixel - buffer.first_pixel + 1
            active_pixels +=
                buffer.pixel_offsets[bucket + 1] > buffer.pixel_offsets[bucket]
        end
        pixels = last_pixel - first_pixel + 1
        if pixels > largest.pixels ||
            (pixels == largest.pixels && active_pixels > largest.active_pixels)
            largest = (ring=ring, pixels=pixels, active_pixels=active_pixels)
        end
    end
    largest.active_pixels > 0 || throw(ArgumentError(
        "the pointing buffer has no samples in any output pixel",
    ))
    return largest
end

function print_memory_estimate(cp, buffer, settings)
    representative = largest_buffer_ring(cp, buffer)
    complete = representative.active_pixels == representative.pixels
    estimate = ring_memory_bytes(
        cp,
        representative.pixels,
        representative.active_pixels,
        1;
        precision=settings.precision,
        complete_ring=complete,
    )
    suggested = suggest_l_tile(
        cp,
        representative.pixels,
        representative.active_pixels;
        precision=settings.precision,
        complete_ring=complete,
        memory_fraction=settings.memory_fraction,
    )
    Printf.format(
        stdout,
        MEMORY_ESTIMATE_FORMAT,
        representative.ring,
        representative.active_pixels,
        representative.pixels,
        estimate.persistent / 2^20,
        estimate.tile_peak / 2^20,
        estimate.total / 2^20,
    )
    @printf(
        "l-tile: Int32 limit %d, suggested %d, configured %s\n",
        max_indexable_l_tile(cp, representative.active_pixels),
        suggested,
        repr(settings.l_tile),
    )
    return nothing
end

function file_sha256(path::AbstractString)
    context = SHA.SHA2_256_CTX()
    buffer = Vector{UInt8}(undef, 1 << 20)
    open(path, "r") do io
        while !eof(io)
            count = readbytes!(io, buffer)
            count == 0 || SHA.update!(context, @view(buffer[1:count]))
        end
    end
    return bytes2hex(SHA.digest!(context))
end

function buffer_sha256(buffer::RingScanBuffer)
    context = SHA.SHA2_256_CTX()
    SHA.update!(context, codeunits(repr((
        buffer.nside,
        buffer.first_ring,
        buffer.last_ring,
        buffer.first_pixel,
        buffer.last_pixel,
        buffer.processed_samples,
        buffer.stored_samples,
    ))))
    for values in (buffer.phi, buffer.theta, buffer.psi, buffer.pixel_offsets)
        SHA.update!(context, reinterpret(UInt8, values))
    end
    return bytes2hex(SHA.digest!(context))
end

function checkpoint_id(settings, alm_hash, blm_hash, pointing_hash)
    descriptor = (
        version=1,
        alm_sha256=alm_hash,
        blm_sha256=blm_hash,
        pointing_sha256=pointing_hash,
        nside=settings.nside,
        lstart=settings.lstart,
        lstop=settings.lstop,
        lmax=settings.lmax,
        bnmax=settings.bnmax,
        kappa=settings.kappa,
        sampling_rate=settings.sampling_rate,
        duration_seconds=settings.duration_seconds,
        chunk_seconds=settings.chunk_seconds,
        precision=string(settings.precision),
        map_precision=string(settings.map_precision),
        beta_order=settings.beta_order,
        beta_margin=settings.beta_margin,
        rcond=settings.rcond,
        moment_method=settings.moment_method,
        l_tile=settings.l_tile,
        memory_fraction=settings.memory_fraction,
    )
    return bytes2hex(SHA.sha256(codeunits(repr(descriptor))))
end

function sphere_progress(ring_total::Int)
    completed = Ref(0)
    next_report = Ref(1)
    interval = max(1, ceil(Int, 0.05 * ring_total))
    return record -> begin
        completed[] += 1
        if completed[] >= next_report[] || completed[] == ring_total
            Printf.format(
                stdout,
                CONVOLUTION_PROGRESS_FORMAT,
                completed[],
                ring_total,
                record.ring,
                record.device,
                record.active_pixels,
                record.hit_count,
                record.elapsed_seconds,
            )
            next_report[] += interval
        end
        return nothing
    end
end

function finite_rms(values)
    finite_values = filter(isfinite, vec(values))
    isempty(finite_values) && return NaN, 0
    return sqrt(mean(abs2, finite_values)), length(finite_values)
end

function write_ring_statistics(path::AbstractString, statistics)
    open(path, "w") do io
        println(io, "ring i_rms finite_values")
        for record in sort(statistics; by=record -> record.ring)
            @printf(io, "%d %.17g %d\n", record.ring, record.i_rms,
                record.finite_values)
        end
    end
    return path
end

function write_metadata(
    path::AbstractString,
    settings,
    cp,
    buffer,
    max_beta,
    beta_max,
    elapsed_seconds,
    identifier,
    alm_hash,
    blm_hash,
    pointing_hash,
    output_path,
)
    gpu_names = join([CUDA.name(device) for device in CUDA.devices()], ", ")
    open(path, "w") do io
        println(io, "run_name = $(settings.run_name)")
        println(io, "created_at = $(Dates.now())")
        println(io, "repo_root = $REPO_ROOT")
        println(io, "julia_version = $(VERSION)")
        println(io, "julia_threads = $(Threads.nthreads())")
        println(io, "gpu_names = $gpu_names")
        println(io, "devices = $(settings.devices)")
        println(io, "checkpoint_id = $identifier")
        println(io, "alm_sha256 = $alm_hash")
        println(io, "blm_sha256 = $blm_hash")
        println(io, "pointing_sha256 = $pointing_hash")
        @printf(io, "max_beta_rad = %.17g\n", max_beta)
        @printf(io, "beta_max_rad = %.17g\n", beta_max)
        @printf(io, "pixel_scale_rad = %.17g\n",
            sqrt(4 * pi / cp.resol.numOfPixels))
        println(io, "ring_count = $(ring_count(cp))")
        println(io, "stored_samples = $(buffer.stored_samples)")
        println(io, "processed_samples = $(buffer.processed_samples)")
        @printf(io, "elapsed_seconds = %.6f\n", elapsed_seconds)
        println(io, "output = $output_path")
        for (name, value) in pairs(settings)
            println(io, "setting_$(name) = $(repr(value))")
        end
        println(io, "cuda_versioninfo:")
        print(io, sprint(CUDA.versioninfo))
    end
    return path
end

function main()
    settings = resolved_settings()
    print_settings(settings)
    alm_teb, blm_teb = preflight!(settings)

    cp = ConvolutionParams(
        nside_output=settings.nside,
        lstart=settings.lstart,
        lstop=settings.lstop,
        lmax=settings.lmax,
        numofsky=1,
        numofbeam=1,
        bnmax=settings.bnmax,
        κ=settings.kappa,
        HWP=false,
        noise=false,
    )
    println("Convolution parameters:")
    show_cp(cp)

    println(
        "Building the pixel-ordered pointing buffer from Falcons chunks...",
    )
    chunks, producer, total_samples = pointing_chunks(settings)
    buffer_started = time()
    buffer = build_ring_scan_buffer(
        cp,
        chunks;
        capacity_hint=total_samples,
        progress_callback=buffer_progress(total_samples),
    )
    fetch(producer)
    Printf.format(
        stdout,
        POINTING_BUFFER_FORMAT,
        length(buffer),
        buffer.processed_samples,
        ring_scan_buffer_bytes(buffer) / 2^20,
        time() - buffer_started,
    )

    max_beta = maximum_buffer_beta(cp, buffer)
    beta_max = settings.beta_margin * max_beta
    beta_max > 0 || throw(ArgumentError(
        "maximum beta is zero; ScanMomentPlan requires a positive beta_max",
    ))
    pixel_scale = sqrt(4 * pi / cp.resol.numOfPixels)
    @printf(
        "max beta = %.6e rad, beta_max = %.6e rad, pixel scale = %.6e rad\n",
        max_beta,
        beta_max,
        pixel_scale,
    )

    scan_plan = ScanMomentPlan(
        kappa=cp.κ,
        nmax=calc_nmax(cp, cp.lstop),
        beta_order=settings.beta_order,
        beta_max=beta_max,
        precision=settings.precision,
    )
    table_started = time()
    table = build_wigner_chebyshev_table_cpu(cp, scan_plan)
    Printf.format(
        stdout,
        WIGNER_TABLE_FORMAT,
        time() - table_started,
        length(table.pair_M),
        table.lstop - table.lstart + 1,
        size(table.coefficients, 1),
    )
    print_memory_estimate(cp, buffer, settings)

    prepared_alm = prepare_alm(cp, alm_teb; precision=settings.precision)
    prepared_blm = prepare_blm(cp, blm_teb; precision=settings.precision)
    alm_hash = file_sha256(settings.alm_path)
    blm_hash = file_sha256(settings.blm_path)
    pointing_hash = buffer_sha256(buffer)
    identifier = checkpoint_id(settings, alm_hash, blm_hash, pointing_hash)
    println("checkpoint_id = $identifier")

    mkpath(settings.output_dir)
    checkpoint_path = settings.output_mode === :full_map ?
        joinpath(settings.output_dir, "$(settings.run_name)_checkpoint.ckpt") :
        nothing
    ring_statistics = NamedTuple[]
    ring_callback = settings.output_mode === :ring_callback ? record -> begin
        i_rms, finite_values = finite_rms(@view(record.stokes[:, :, 1, :]))
        push!(ring_statistics, (
            ring=record.ring,
            i_rms=i_rms,
            finite_values=finite_values,
        ))
        return nothing
    end : nothing

    println("Starting full-sphere convolution...")
    started = time()
    maps = convolve_sphere_multigpu(
        cp,
        buffer,
        prepared_alm,
        prepared_blm,
        table,
        scan_plan;
        devices=settings.devices,
        output=settings.output_mode,
        ring_callback=ring_callback,
        progress_callback=sphere_progress(ring_count(cp)),
        reclaim_interval=settings.reclaim_interval,
        checkpoint_path=checkpoint_path,
        checkpoint_id=checkpoint_path === nothing ? nothing : identifier,
        checkpoint_interval=checkpoint_path === nothing ? 0 :
            settings.checkpoint_interval,
        resume=settings.resume,
        l_tile=settings.l_tile,
        memory_fraction=settings.memory_fraction,
        moment_method=settings.moment_method,
        rcond=settings.rcond,
        unseen=NaN,
        map_precision=settings.map_precision,
    )
    elapsed_seconds = time() - started

    output_path = "none"
    if settings.output_mode === :full_map
        output_path = joinpath(
            settings.output_dir,
            "$(settings.run_name)_maps.npy",
        )
        npzwrite(output_path, maps)
        println("Saved maps to $output_path")
        for (index, name) in enumerate(("I", "Q", "U"))
            component_rms, finite_values =
                finite_rms(@view(maps[:, :, index, :]))
            @printf("%s rms = %.6g (%d finite values)\n", name,
                component_rms, finite_values)
        end
    elseif settings.output_mode === :ring_callback
        output_path = joinpath(
            settings.output_dir,
            "$(settings.run_name)_ring_i_rms.txt",
        )
        write_ring_statistics(output_path, ring_statistics)
        println("Saved ring I rms statistics to $output_path")
    else
        println("OUTPUT_MODE=:discard; no map or ring statistics were saved")
    end

    meta_path = joinpath(settings.output_dir, "$(settings.run_name)_meta.txt")
    write_metadata(
        meta_path,
        settings,
        cp,
        buffer,
        max_beta,
        beta_max,
        elapsed_seconds,
        identifier,
        alm_hash,
        blm_hash,
        pointing_hash,
        output_path,
    )
    @printf("Completed in %.1f s. Metadata: %s\n", elapsed_seconds, meta_path)
    return nothing
end

main()
