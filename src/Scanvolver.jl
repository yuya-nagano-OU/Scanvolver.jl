module Scanvolver

using CUDA
using Healpix: Resolution, pix2angRing, ang2pixRing, getringinfo,
    numOfRings
using LinearAlgebra: cond, lu
using SHA: sha256
using SpecialFunctions: loggamma
using StaticArrays: @SMatrix

include("rotation.jl")
include("parameters.jl")
include("angles.jl")
include("harmonics/alm.jl")
include("harmonics/alm_gpu.jl")
include("harmonics/blm.jl")
include("scan_moment_plan.jl")
include("scan_moments/scan_moments_cpu.jl")
include("scan_moments/scan_moments_direct_gpu.jl")
include("scan_moments/scan_moments_gemm_gpu.jl")
include("wigner/truncation.jl")
include("wigner/wigner_seed.jl")
include("wigner/ringd_gpu.jl")
include("wigner/wigner_table_gpu.jl")
include("wigner/wigner_table_cpu.jl")
include("effective_beam/effective_beam.jl")
include("sky_ring/sky_ring.jl")
include("sky_ring/active_pixels.jl")
include("geometry/ring_geometry.jl")
include("geometry/scan_buffer.jl")
include("convolution/memory_model.jl")
include("convolution/contraction.jl")
include("convolution/convolve_ring.jl")
include("convolution/checkpoint.jl")
include("convolution/convolve_sphere.jl")
include("mapmaking/mapmaking.jl")

export ConvolutionParams, Angles, show_cp
export PreparedAlm, prepare_alm, alm_slice
export GPUAlm, upload_alm_gpu
export PreparedBlm, prepare_blm, blm_slice
export ScanMomentPlan, chebyshev_nodes
export scan_moments_cpu, scan_moments_gpu
export scan_moments_direct_gpu, scan_moments_gemm_gpu
export scan_moments_gemm_memory_bytes
export calc_Mmax, calc_nmax
export ringd_blocks_recursion_gpu, ringd_blocks_recursion_gpu!
export RingDContext, ringd_context, free_ringd_context!
export WignerChebyshevTable
export build_wigner_chebyshev_table_cpu
export upload_wigner_table_gpu
export build_effective_kernel_gpu, apply_beams_gemm_gpu
export build_effective_beam_gpu
export build_sky_ring_gpu
export ActiveRingPixels, select_active_sky_ring_gpu
export free_active_sky_ring!
export RingGeometry, ring_geometry, ring_pixel_range, ring_count
export make_ring_angles
export RingScanBuffer, build_ring_scan_buffer
export ring_scan_buffer_bytes, ring_hit_count
export accumulate_convolution_tile_gemm_gpu!
export ring_memory_bytes, max_indexable_l_tile, suggest_l_tile
export RingConvolution, convolve_ring_gpu, free_ring_convolution!
export convolve_sphere_gpu, convolve_sphere_multigpu
export SphereCheckpoint, save_sphere_checkpoint, load_sphere_checkpoint
export allocate_stokes_map, binning_mapmake!, ring_stokes

end
