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
