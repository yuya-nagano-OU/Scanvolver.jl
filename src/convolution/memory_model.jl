"""
    ring_memory_bytes(
        cp, ring_pixel_count, active_count, l_tile;
        precision=Float64, complete_ring=false,
    ) -> NamedTuple

Device bytes that one ring of the convolution needs, split into the parts
that live for the whole ring and the parts that live for one `l` tile.

Fields: `ringd_plan`, `ringd`, `accumulated`, `persistent`;
`effective_kernel`, `beam_workspace`, `effective_beam`, `sky_ring`,
`active_sky_ring`, `tile_peak`; and `total = persistent + tile_peak`.

Scan moments are **not** included: they are already resident when the tiles are
sized, and `scan_moments_gemm_memory_bytes` estimates them separately. Neither
is the cuFFT workspace of the ring transform, which is plan-dependent and is
the one term here that is not a closed form.

`tile_peak` is the larger of the two moments inside a tile: building the
effective beam, where the compact kernel, the beam matrices, the beam GEMM
output and the effective beam are all live at once, and contracting, where the
effective beam, the full-ring sky tile and its gathered copy are live. The
gathered copy is free on a complete ring, where `select_active_sky_ring_gpu`
returns its input.
"""
function ring_memory_bytes(
    cp::ConvolutionParams,
    ring_pixel_count::Integer,
    active_count::Integer,
    l_tile::Integer;
    precision::Type=Float64,
    complete_ring::Bool=false,
)
    precision in (Float32, Float64) || throw(ArgumentError(
        "precision must be Float32 or Float64, got $precision",
    ))
    ring_pixels = Int(ring_pixel_count)
    actives = Int(active_count)
    tile = Int(l_tile)
    ring_pixels > 0 || throw(ArgumentError(
        "ring_pixel_count must be positive, got $ring_pixels",
    ))
    0 < actives <= ring_pixels || throw(ArgumentError(
        "active_count must be in 1:$ring_pixels, got $actives",
    ))
    tile >= 1 || throw(ArgumentError("l_tile must be positive, got $tile"))

    complex_bytes = Int128(2 * sizeof(precision))
    real_bytes = Int128(sizeof(precision))
    int32_bytes = Int128(4)

    Mmax = calc_Mmax(cp, cp.lstop)
    nmax = calc_nmax(cp, cp.lstop)
    M_count = Int128(2 * Mmax + 1)
    delta_count = Int128(2 * cp.κ + 1)
    term_count = Int128(3) * M_count * tile
    batch_count = M_count * tile
    beams = Int128(cp.numofbeam)
    skies = Int128(cp.numofsky)
    l_count = Int128(cp.lstop - cp.lstart + 1)

    # One seed per (m, M): M = 0 contributes m = 0:lstop, every other M the
    # full -lstop:lstop.
    pair_count =
        Int128(cp.lstop + 1) + Int128(Mmax) * Int128(2 * cp.lstop + 1)
    ringd_plan = pair_count * (Int128(5) * int32_bytes + Int128(2) * real_bytes)
    ringd = Int128(2 * cp.lstop + 1) * M_count * l_count * complex_bytes
    accumulated = skies * beams * Int128(actives) * 2 * complex_bytes
    persistent = ringd_plan + ringd + accumulated

    effective_kernel =
        delta_count * Int128(actives) * batch_count * 2 * complex_bytes
    beam_matrices = Int128(3) * beams * delta_count * batch_count * complex_bytes
    gemm_output = Int128(3) * beams * Int128(actives) * batch_count * complex_bytes
    packed_blm =
        Int128(2 * nmax + 1) * 3 * beams * Int128(tile) * complex_bytes
    effective_beam = term_count * beams * Int128(actives) * 2 * complex_bytes
    beam_workspace = max(packed_blm, beam_matrices + gemm_output)

    sky_ring = skies * term_count * Int128(ring_pixels) * complex_bytes
    active_sky_ring = complete_ring ?
        Int128(0) :
        skies * term_count * Int128(actives) * complex_bytes

    build_peak = effective_kernel + beam_workspace + effective_beam
    contract_peak = effective_beam + sky_ring + active_sky_ring
    tile_peak = max(build_peak, contract_peak)

    return (
        ringd_plan=ringd_plan,
        ringd=ringd,
        accumulated=accumulated,
        persistent=persistent,
        effective_kernel=effective_kernel,
        beam_workspace=beam_workspace,
        effective_beam=effective_beam,
        sky_ring=sky_ring,
        active_sky_ring=active_sky_ring,
        tile_peak=tile_peak,
        total=persistent + tile_peak,
    )
end

"""
    max_indexable_l_tile(cp, active_count) -> Int

Largest `l` tile whose kernels still index with `Int32`.

`build_effective_kernel_gpu` and `apply_beams_gemm_gpu` reject a tile whose
flattened extent exceeds `typemax(Int32)`, so this is a hard ceiling
independent of how much memory is free. Returns 0 when even a single `l` is
too wide, which means the ring itself has to be split by pixels.
"""
function max_indexable_l_tile(
    cp::ConvolutionParams,
    active_count::Integer,
)
    actives = Int128(active_count)
    actives > 0 || throw(ArgumentError(
        "active_count must be positive, got $active_count",
    ))
    Mmax = calc_Mmax(cp, cp.lstop)
    M_count = Int128(2 * Mmax + 1)
    delta_count = Int128(2 * cp.κ + 1)
    beams = Int128(cp.numofbeam)
    limit = Int128(typemax(Int32))

    # Per unit of l tile, from the four Int32-indexed launches.
    per_tile = max(
        Int128(2) * delta_count * actives * M_count,   # compact kernel
        Int128(3) * beams * delta_count * M_count,     # beam GEMM matrices
        Int128(2) * 3 * M_count * beams * actives,     # effective beam tile
        Int128(3) * beams * actives * M_count,         # beam GEMM output
    )
    per_tile == 0 && return cp.lstop - cp.lstart + 1
    return Int(min(Int128(cp.lstop - cp.lstart + 1), limit ÷ per_tile))
end

"""Free device memory, counting memory CUDA.jl has cached for reuse."""
_available_device_bytes() = min(Int128(CUDA.total_memory()),
                                Int128(CUDA.free_memory()) + Int128(CUDA.cached_memory()))

"""
    streamed_tile_bytes(cp_batch, ring_pixels, active_pixels, l_tile; precision, complete_ring)

Peak explicit device buffers for one tile, including simultaneous K + full U
+ gathered U during beam application. K and U both remain resident across beam
batches. Scan moments, ringD, accumulators,
the uploaded alm/table and cuFFT's plan-dependent workspace are excluded.
"""
function streamed_tile_bytes(cp, ring_pixels, active_pixels, tile;
                             precision=Float32, complete_ring=false)
    sizes = ring_memory_bytes(cp, ring_pixels, active_pixels, tile;
                              precision, complete_ring)
    bytes = Int128(sizeof(Complex{precision}))
    modes = Int128(2calc_Mmax(cp, cp.lstop) + 1)
    delta = Int128(2cp.κ + 1)
    beams = Int128(cp.numofbeam)
    packed = Int128(2calc_nmax(cp, cp.lstop) + 1) * 3 * beams * tile * bytes
    matrices = 3 * beams * delta * modes * tile * bytes
    gemm = 3 * beams * Int128(active_pixels) * modes * tile * bytes
    beam_peak = max(packed + matrices, matrices + gemm + sizes.effective_beam)
    return sizes.effective_kernel + sizes.sky_ring + sizes.active_sky_ring + beam_peak
end

# What `memory_fraction=:auto` leaves unused: the explicit buffers and the cuFFT
# workspace are both counted then, so this only covers allocator granularity.
const _AUTO_MEMORY_FRACTION = 0.95
# A cuFFT plan measures about 10 MiB above cufftEstimateMany, whatever its size.
const _CUFFT_PLAN_OVERHEAD_BYTES = Int128(16) * 2^20

"""
`(fraction, count_fft)` of a `memory_fraction` setting. A number is the share of
the available device memory the explicit tile buffers may take, with the cuFFT
workspace left to the remainder. `:auto` counts the cuFFT workspace as well and
uses all but a small reserve.
"""
function resolve_memory_fraction(setting)
    setting === :auto && return _AUTO_MEMORY_FRACTION, true
    setting isa Real && isfinite(setting) && 0 < setting <= 1 ||
        throw(ArgumentError("memory_fraction must be :auto or in (0, 1], got $setting"))
    return Float64(setting), false
end

# Measured on an RTX A6000 (see plan_beam_batches): batches of about 20 beams are
# as fast as larger ones, and tiles wider than 8 multipoles are no faster.
const _AUTO_BEAM_BATCH = 20
const _AUTO_MAX_L_TILE = 8

"""Upper limit of an automatic `l` tile: `_AUTO_MAX_L_TILE` for `:auto`."""
function resolve_max_l_tile(setting, l_count::Integer)
    setting === :auto && return min(Int(l_count), _AUTO_MAX_L_TILE)
    setting isa Integer && setting > 0 ||
        throw(ArgumentError("max_l_tile must be :auto or a positive integer, got $setting"))
    return Int(setting)
end

"""
    sky_ring_fft_workspace_bytes(precision, sky_count, term_count, pixel_count) -> Int128

Device bytes cuFFT allocates for the in-place inverse transform along the pixel
axis of `U[sky, term, pixel]`, asked of cuFFT itself (`cufftEstimateMany`, the
strided batched layout `build_sky_ring_gpu` plans). It is either nothing, on
ring lengths cuFFT handles directly, or one more copy of `U`; on an RTX A6000
the measured plan allocation is this value plus about 10 MiB, which
`_CUFFT_PLAN_OVERHEAD_BYTES` covers. Needs a functional CUDA device.
"""
function sky_ring_fft_workspace_bytes(precision::Type, sky_count::Integer,
                                      term_count::Integer, pixel_count::Integer)
    batch = Int128(sky_count) * Int128(term_count)
    array_bytes = batch * pixel_count * 2 * sizeof(precision)
    # Beyond the 32-bit estimate interface, assume the larger of the two cases.
    max(batch, pixel_count) <= typemax(Cint) || return array_bytes
    workspace = Ref{Csize_t}(0)
    lengths = Cint[pixel_count]
    kind = precision === Float32 ? CUDA.CUFFT.CUFFT_C2C : CUDA.CUFFT.CUFFT_Z2Z
    CUDA.CUFFT.cufftEstimateMany(1, lengths, lengths, Cint(batch), Cint(1), lengths,
                                 Cint(batch), Cint(1), kind, Cint(batch), workspace)
    return Int128(workspace[])
end

function select_tile(cp_batch, geometry, active, T, l_tile, max_l_tile, fraction;
                     count_fft::Bool=false)
    unit = streamed_tile_bytes(cp_batch, geometry.pixel_count, length(active), 1;
                               precision=T, complete_ring=active.complete)
    budget = floor(Int128, fraction * Float64(_available_device_bytes()))
    if count_fft
        terms = 3 * (2 * calc_Mmax(cp_batch, cp_batch.lstop) + 1)
        unit += sky_ring_fft_workspace_bytes(T, cp_batch.numofsky, terms, geometry.pixel_count)
        budget -= _CUFFT_PLAN_OVERHEAD_BYTES
    end
    index_limit = max_indexable_l_tile(cp_batch, length(active))
    ceiling = min(cp_batch.lstop - cp_batch.lstart + 1, index_limit)
    width = l_tile === :auto ?
        Int(min(Int128(min(ceiling, max_l_tile)), budget ÷ unit)) :
        min(Int(l_tile), cp_batch.lstop - cp_batch.lstart + 1)
    width >= 1 && width <= ceiling && unit * width <= budget ||
        throw(ArgumentError("ring $(geometry.ring): l tile does not fit the device/index budget " *
                            "(one-l bytes=$unit, budget=$budget, index_limit=$index_limit)"))
    return width, unit * width
end

"""
    resident_ring_bytes(cp, precision; beta_order=8) -> Int128

Device bytes that stay allocated while the `l` tiles of the widest ring
(`4 * nside_output` pixels, all hit) are processed: the uploaded skies and
Wigner-Chebyshev table, the ringD plan and stack, the scan moments and the
accumulators of every beam. The table term is an upper bound, since it counts
every `(M, n)` pair of the `kappa` band at every `l`.
"""
function resident_ring_bytes(cp::ConvolutionParams, precision::Type; beta_order::Integer=8)
    pixels = 4 * cp.nside_output
    complex_bytes = Int128(2 * sizeof(precision))
    l_count = Int128(cp.lstop - cp.lstart + 1)
    n_count = Int128(2 * calc_nmax(cp, cp.lstop) + 1)
    M_count = Int128(2 * calc_Mmax(cp, cp.lstop) + 1)
    delta_count = Int128(2 * cp.κ + 1)
    ring = ring_memory_bytes(cp, pixels, pixels, 1; precision)
    moments = Int128(2) * beta_order * delta_count * n_count * pixels * complex_bytes
    alm = Int128(2 * cp.lstop + 1) * 3 * cp.numofsky * l_count * complex_bytes
    table = Int128(beta_order) * delta_count * n_count * l_count * sizeof(precision) +
            M_count * n_count * sizeof(Int32)
    return ring.persistent + moments + alm + table
end

"""
    plan_beam_batches(cp, precision; available_bytes, memory_fraction=:auto,
                      l_tile=:auto, max_l_tile=:auto, beta_order=8, multiple=1)

Choose how many beams to apply together, from the memory model alone.

The plan is made for the widest ring (`4 * nside_output` pixels, all hit, with
the gathered sky copy of an incomplete ring), so it holds for every other ring.
`available_bytes` is the device memory free before anything is uploaded;
`resident_ring_bytes` is taken off it and `memory_fraction` of the rest is the
tile budget, the same rule `l_tile=:auto` applies ring by ring. With
`memory_fraction=:auto` each multipole of a tile is charged one extra copy of
the sky tile for the cuFFT workspace, the larger of the two cases
`sky_ring_fft_workspace_bytes` can return, so the plan needs no device.

The rule follows a timing sweep, not the memory limit (RTX A6000, Float32,
nside 512, lmax 1200, bnmax 120, kappa 10, 10 skies, 40 beams, two full rings):
going from 5 to 10 beams per batch made a ring about 22% faster and 10 to 20
another 3 to 4%, with nothing more at 40; the `l` tile width changed the time
by 1 to 2% between 1 and 8, and wider tiles were up to 15% slower with 5 skies
and 5 beams. So the beams are split into the fewest batches of about
`_AUTO_BEAM_BATCH` each, in whole multiples of `multiple` (the detectors of one
group, or 1), the batch is reduced only if it does not fit, and the tile is the
widest that fits the budget, the `Int32` index limit and `max_l_tile`, whose
`:auto` value is `_AUTO_MAX_L_TILE`. Memory beyond that is left unused because
it buys no speed.

Returns `(batch_size, l_tile, batch_count, tile_count, tile_bytes,
resident_bytes, tile_budget)`. Throws `ArgumentError` when not even one group
of beams and one multipole fit.
"""
function plan_beam_batches(cp::ConvolutionParams, precision::Type;
                           available_bytes::Integer, memory_fraction=:auto,
                           l_tile=:auto, max_l_tile=:auto,
                           beta_order::Integer=8, multiple::Integer=1)
    precision in (Float32, Float64) || throw(ArgumentError(
        "precision must be Float32 or Float64, got $precision"))
    fraction, count_fft = resolve_memory_fraction(memory_fraction)
    l_tile === :auto || (l_tile isa Integer && l_tile > 0) || throw(ArgumentError("invalid l_tile"))
    tile_cap = resolve_max_l_tile(max_l_tile, cp.lstop - cp.lstart + 1)
    multiple > 0 && cp.numofbeam % multiple == 0 || throw(ArgumentError(
        "$(cp.numofbeam) beams cannot be divided into groups of $multiple"))

    pixels = 4 * cp.nside_output
    l_count = cp.lstop - cp.lstart + 1
    resident = resident_ring_bytes(cp, precision; beta_order)
    budget = floor(Int128, fraction * Float64(max(Int128(available_bytes) - resident, 0)))
    count_fft && (budget -= _CUFFT_PLAN_OVERHEAD_BYTES)

    target = cld(cp.numofbeam, cld(cp.numofbeam, _AUTO_BEAM_BATCH))
    preferred = min(cp.numofbeam, cld(target, multiple) * multiple)
    best = nothing
    for batch_size in preferred:-multiple:multiple
        cp_batch = ConvolutionParams(cp; numofbeam=batch_size)
        unit = streamed_tile_bytes(cp_batch, pixels, pixels, 1; precision)
        count_fft && (unit += ring_memory_bytes(cp_batch, pixels, pixels, 1; precision).sky_ring)
        ceiling = min(l_count, max_indexable_l_tile(cp_batch, pixels))
        width = l_tile === :auto ?
            Int(min(Int128(min(ceiling, tile_cap)), max(budget, 0) ÷ unit)) :
            min(Int(l_tile), l_count)
        1 <= width <= ceiling && unit * width <= budget || continue
        best = (; batch_size, l_tile=width, batch_count=cld(cp.numofbeam, batch_size),
                tile_count=cld(l_count, width), tile_bytes=unit * width,
                resident_bytes=resident, tile_budget=budget)
        break
    end
    best === nothing && throw(ArgumentError(
        "no beam batch fits the device: available=$(available_bytes) bytes, " *
        "resident=$resident, tile budget=$budget; reduce the sky or beam count, " *
        "kappa or l_tile, or use a larger GPU"))
    return best
end
