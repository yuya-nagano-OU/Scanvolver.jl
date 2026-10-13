const _SKY_RING_THREADS = 256

mutable struct _SkyRingFFTPlanCache
    plans::Dict{Tuple{Vararg{Int}},Any}
end

_SkyRingFFTPlanCache() = _SkyRingFFTPlanCache(
    Dict{Tuple{Vararg{Int}},Any}(),
)

function _sky_ring_ifft!(sky_ring, cache::_SkyRingFFTPlanCache)
    key = Tuple(size(sky_ring))
    fft_plan = get!(cache.plans, key) do
        CUDA.CUFFT.plan_ifft!(sky_ring, (3,))
    end
    fft_plan * sky_ring
    return sky_ring
end

function _release_sky_ring_fft_plans!(cache::_SkyRingFFTPlanCache)
    CUDA.synchronize()
    for fft_plan in values(cache.plans)
        finalize(fft_plan.p)
    end
    empty!(cache.plans)
    # Finalizing a plan returns its native handle to cuFFT.jl's global cache.
    # The cache key contains the complete input shape, so polar rings would
    # otherwise retain a separate workspace for every pixel count.  cuFFT.jl
    # 6.3.1 does not register this cache with CUDA.reclaim(), hence the
    # explicit purge after every ring.
    empty!(CUDA.CUFFT.idle_handles)
    return nothing
end

@inline function _sky_ring_phase(angle::T) where {T<:_HARMONIC_PRECISIONS}
    return Complex{T}(cos(angle), sin(angle))
end

function _check_ring_longitudes(phi_pixels::AbstractVector{<:Real})
    pixel_count = length(phi_pixels)
    pixel_count > 0 || throw(ArgumentError(
        "phi_pixels must be nonempty",
    ))
    phi0 = Float64(first(phi_pixels))
    step = 2pi / pixel_count
    tolerance = 2e-10
    @inbounds for pixel_index in eachindex(phi_pixels)
        expected = phi0 + (pixel_index - 1) * step
        phase_error = abs(
            cis(Float64(phi_pixels[pixel_index]) - expected) - 1,
        )
        phase_error <= tolerance || throw(ArgumentError(
            "phi_pixels are not an equally spaced ascending ring",
        ))
    end
    return phi0
end

function _check_sky_ring_inputs(
    cp::ConvolutionParams,
    ringd::CuArray,
    alm::GPUAlm,
    phi_pixels::AbstractVector{<:Real},
    l_first::Int,
    l_last::Int,
)
    cp.HWP == false || throw(ArgumentError(
        "build_sky_ring_gpu currently supports HWP=false",
    ))
    cp.lstart == alm.lstart &&
        cp.lstop == alm.lstop &&
        cp.lmax == alm.lmax &&
        cp.numofsky == alm.numofsky || throw(ArgumentError(
        "GPUAlm is incompatible with ConvolutionParams",
    ))
    cp.lstart <= l_first <= l_last <= cp.lstop ||
        throw(ArgumentError(
            "l tile must lie in $(cp.lstart):$(cp.lstop)",
        ))
    expected_ringd = (
        2 * cp.lstop + 1,
        2 * calc_Mmax(cp, cp.lstop) + 1,
        cp.lstop - cp.lstart + 1,
    )
    size(ringd) == expected_ringd || throw(DimensionMismatch(
        "ringd has size $(size(ringd)); expected $expected_ringd",
    ))
    length(phi_pixels) <= typemax(Int32) || throw(ArgumentError(
        "ring pixel count must fit in Int32",
    ))
    return _check_ring_longitudes(phi_pixels)
end

function _build_sky_ring_spectrum_kernel!(
    sky_ring,
    ringd,
    alm,
    phi0::T,
    pixel_count::Int32,
    Mmax_global::Int32,
    l_first::Int32,
    lstart::Int32,
    tile_count::Int32,
    sky_count::Int32,
) where {T<:_HARMONIC_PRECISIONS}
    linear_index =
        (Int64(blockIdx().x) - Int64(1)) * Int64(blockDim().x) +
        Int64(threadIdx().x)
    M_count = Int32(2) * Mmax_global + Int32(1)
    term_count = Int64(3) * Int64(M_count) * Int64(tile_count)
    total = Int64(sky_count) * term_count * Int64(pixel_count)
    linear_index > total && return nothing
    index = linear_index - Int64(1)
    sky_index = Int32(index % Int64(sky_count) + Int64(1))
    index ÷= Int64(sky_count)
    term_index = index % term_count + Int64(1)
    frequency_index = Int32(index ÷ term_count + Int64(1))

    term_offset = term_index - Int64(1)
    spin_index = Int32(term_offset % Int64(3) + Int64(1))
    term_offset ÷= Int64(3)
    M_index = Int32(term_offset % Int64(M_count) + Int64(1))
    tile_index = Int32(term_offset ÷ Int64(M_count) + Int64(1))

    l = l_first + tile_index - Int32(1)
    l_index = l - lstart + Int32(1)
    M = M_index - Mmax_global - Int32(1)
    Mmax_l = min(Mmax_global, l)
    abs(M) <= Mmax_l || return nothing
    local_M_index = M + Mmax_l + Int32(1)

    target_remainder = frequency_index - Int32(1)
    first_m = -l
    first_remainder = mod(first_m, pixel_count)
    m = first_m + mod(
        target_remainder - first_remainder,
        pixel_count,
    )
    value = zero(eltype(sky_ring))
    while m <= l
        m_index = m + l + Int32(1)
        @inbounds value +=
            ringd[m_index, local_M_index, l_index] *
            alm[m_index, spin_index, sky_index, l_index] *
            _sky_ring_phase(T(m) * phi0)
        m += pixel_count
    end
    spin_weight = spin_index == Int32(1) ? one(T) : T(0.5)
    @inbounds sky_ring[sky_index, term_index, frequency_index] =
        T(pixel_count) * spin_weight * value
    return nothing
end

"""
    build_sky_ring_gpu(
        cp, ringd, alm_gpu, phi_pixels, l_first, l_last,
    )

Evaluate one `l` tile of the sky/ringD contraction on an equally spaced
ascending ring:

```math
U_{i,(s,M,l),p}=w_s \\sum_m
d^l_{mM}(\\theta_{ring})a^s_{i,lm}e^{i m \\phi_p},
\\qquad w_s=(1,1/2,1/2).
```

The inverse FFT is performed directly in the final GEMM layout
`U[sky, (spin,M,l), pixel]`. Spin varies fastest in `(0,+2,-2)` order,
followed by signed `M`, then `l`. Modes that alias on a short polar ring are
summed into the same FFT bin exactly; no post-FFT transpose is required.
"""
function build_sky_ring_gpu(
    cp::ConvolutionParams,
    ringd::CuArray{Complex{T},3},
    alm::GPUAlm{T},
    phi_pixels::AbstractVector{<:Real},
    l_first::Integer,
    l_last::Integer;
    plan_cache::Union{Nothing,_SkyRingFFTPlanCache}=nothing,
) where {T<:_HARMONIC_PRECISIONS}
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    first_l = Int(l_first)
    last_l = Int(l_last)
    phi0 = T(_check_sky_ring_inputs(
        cp,
        ringd,
        alm,
        phi_pixels,
        first_l,
        last_l,
    ))

    pixel_count = length(phi_pixels)
    Mmax_global = calc_Mmax(cp, cp.lstop)
    M_count = 2 * Mmax_global + 1
    tile_count = last_l - first_l + 1
    term_count = 3 * M_count * tile_count
    total =
        Int128(alm.numofsky) * term_count * pixel_count
    total <= typemax(Int64) || throw(ArgumentError(
        "sky-ring tile has $total elements, exceeding Int64 indexing; " *
        "reduce the l tile",
    ))

    sky_ring = CUDA.zeros(
        Complex{T},
        alm.numofsky,
        term_count,
        pixel_count,
    )
    fft_plan = nothing
    succeeded = false
    try
        @cuda threads=_SKY_RING_THREADS blocks=cld(
            Int(total),
            _SKY_RING_THREADS,
        ) _build_sky_ring_spectrum_kernel!(
            sky_ring,
            ringd,
            alm.values,
            phi0,
            Int32(pixel_count),
            Int32(Mmax_global),
            Int32(first_l),
            Int32(cp.lstart),
            Int32(tile_count),
            Int32(alm.numofsky),
        )
        if plan_cache === nothing
            fft_plan = CUDA.CUFFT.plan_ifft!(sky_ring, (3,))
            fft_plan * sky_ring
        else
            _sky_ring_ifft!(sky_ring, plan_cache)
        end
        CUDA.synchronize()
        succeeded = true
        return sky_ring
    finally
        if fft_plan !== nothing
            CUDA.synchronize()
            finalize(fft_plan.p)
            # `finalize` only makes the native handle idle.  A standalone
            # caller has no ring cache whose release would destroy it.
            empty!(CUDA.CUFFT.idle_handles)
        end
        succeeded || CUDA.unsafe_free!(sky_ring)
    end
end
