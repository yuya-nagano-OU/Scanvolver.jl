# Real Stokes skies have only one independent polarization helicity.  The
# scalar ring modes also obey U0(-M)=(-1)^M*conj(U0(M)).  This representation
# stores precisely 3*(2*Mmax+1) real values per l, half the old complex tile.
struct _SpinReducedSkyRing{T<:_HARMONIC_PRECISIONS}
    values::CuArray{T,3}
    Mmax::Int
    l_count::Int
end
Base.size(s::_SpinReducedSkyRing) = size(s.values)
Base.size(s::_SpinReducedSkyRing, d::Integer) = size(s.values, d)
Base.eltype(::_SpinReducedSkyRing{T}) where {T} = T
Base.length(s::_SpinReducedSkyRing) = length(s.values)
Base.Array(s::_SpinReducedSkyRing) = Array(s.values)
CUDA.unsafe_free!(s::_SpinReducedSkyRing) = CUDA.unsafe_free!(s.values)

function _alm_has_spin_reality(alm::PreparedAlm)
    for l in alm.lstart:alm.lstop
        a = alm_slice(alm, l)
        for m in 0:l, sky in 1:alm.numofsky
            p, n = m+l+1, -m+l+1
            sign = isodd(m) ? -1 : 1
            a[1][sky,n] == sign*conj(a[1][sky,p]) || return false
            a[2][sky,n] == sign*conj(a[3][sky,p]) || return false
            a[3][sky,n] == sign*conj(a[2][sky,p]) || return false
        end
    end
    return true
end

function _resolve_spin_reduction(alm::PreparedAlm, mode::Symbol)
    mode in (:auto, :on, :off) ||
        throw(ArgumentError("spin_reduction must be :auto, :on or :off"))
    mode === :off && return false
    valid = _alm_has_spin_reality(alm)
    mode === :on && !valid && throw(ArgumentError(
        "spin_reduction=:on requires real-Stokes sky coefficients, including m=0"))
    return valid
end

function select_active_sky_ring_gpu(sky::_SpinReducedSkyRing, active::ActiveRingPixels)
    values = _select_active_sky_ring_values(sky.values, active)
    values === sky.values && return sky
    return _SpinReducedSkyRing(values, sky.Mmax, sky.l_count)
end

function free_active_sky_ring!(active::_SpinReducedSkyRing, sky::_SpinReducedSkyRing)
    active === sky || CUDA.unsafe_free!(active)
    return nothing
end

function _spin_reduced_spectrum_kernel!(spectrum, ringd, alm, phi0::T,
        pixel_count::Int32, G::Int32, l_first::Int32, lstart::Int32,
        tile_count::Int32, sky_count::Int32) where {T}
    linear = (Int64(blockIdx().x)-1)*Int64(blockDim().x)+Int64(threadIdx().x)
    linear > length(spectrum) && return
    index = linear-1
    sky = Int32(index % sky_count + 1); index ÷= sky_count
    fft_count = Int64(3G+2)
    term = index % (fft_count*tile_count)
    frequency = Int32(index ÷ (fft_count*tile_count))
    local_term = Int32(term % fft_count)
    tile = Int32(term ÷ fft_count)
    scalar = local_term <= G
    spin = scalar ? Int32(1) : Int32(2)
    M = scalar ? local_term : local_term-(G+1)-G
    l = l_first+tile
    abs(M) <= min(G,l) || return
    local_M = M+min(G,l)+1
    first_m = -l
    m = first_m+mod(frequency-mod(first_m,pixel_count),pixel_count)
    value = zero(eltype(spectrum))
    while m <= l
        @inbounds value += ringd[m+l+1,local_M,l-lstart+1] *
            alm[m+l+1,spin,sky,l-lstart+1] * _sky_ring_phase(T(m)*phi0)
        m += pixel_count
    end
    weight = scalar ? one(T) : T(0.5)
    @inbounds spectrum[linear] = T(pixel_count)*weight*value
    return
end

function _pack_spin_reduced_sky_kernel!(output, spectrum, G::Int32,
        tile_count::Int32, sky_count::Int32)
    linear = (Int64(blockIdx().x)-1)*Int64(blockDim().x)+Int64(threadIdx().x)
    linear > length(output) && return
    index = linear-1
    sky = Int32(index % sky_count+1); index ÷= sky_count
    Mc = Int64(2G+1)
    terms = 3Mc*tile_count
    term = index % terms
    pixel = index ÷ terms+1
    tile = term ÷ (3Mc)
    t = term % (3Mc)
    if t < Mc
        # [U0(0), Re U0(1), Im U0(1), ..., Re U0(G), Im U0(G)]
        M = (t+1)÷2
        fft_term = tile*(3G+2)+M+1
        imaginary = t > 0 && iseven(t)
    else
        # [Re U+(-G), Im U+(-G), ..., Re U+(G), Im U+(G)]
        p = t-Mc
        fft_term = tile*(3G+2)+(G+1)+p÷2+1
        imaginary = isodd(p)
    end
    @inbounds value = spectrum[sky,fft_term,pixel]
    @inbounds output[linear] = imaginary ? imag(value) : real(value)
    return
end

function _build_spin_reduced_sky_ring(cp, ringd, alm::GPUAlm{T}, phi0,
        pixel_count, l_first, l_last, plan_cache) where {T}
    G = calc_Mmax(cp,cp.lstop)
    tiles = l_last-l_first+1
    count = Int128(alm.numofsky)*(3G+2)*tiles*pixel_count
    count <= typemax(Int64) || throw(ArgumentError("spin-reduced FFT tile exceeds Int64"))
    spectrum = output = fft_plan = nothing
    succeeded = false
    try
        spectrum = CUDA.zeros(Complex{T},alm.numofsky,(3G+2)*tiles,pixel_count)
        @cuda threads=_SKY_RING_THREADS blocks=cld(length(spectrum),_SKY_RING_THREADS) _spin_reduced_spectrum_kernel!(
            spectrum,ringd,alm.values,T(phi0),Int32(pixel_count),Int32(G),
            Int32(l_first),Int32(cp.lstart),Int32(tiles),Int32(alm.numofsky))
        if plan_cache === nothing
            fft_plan = CUDA.CUFFT.plan_ifft!(spectrum,(3,))
            fft_plan*spectrum
        else
            _sky_ring_ifft!(spectrum,plan_cache)
        end
        output = CUDA.zeros(T,alm.numofsky,3*(2G+1)*tiles,pixel_count)
        @cuda threads=_SKY_RING_THREADS blocks=cld(length(output),_SKY_RING_THREADS) _pack_spin_reduced_sky_kernel!(
            output,spectrum,Int32(G),Int32(tiles),Int32(alm.numofsky))
        CUDA.synchronize()
        succeeded = true
        return _SpinReducedSkyRing(output,G,tiles)
    finally
        CUDA.synchronize()
        spectrum === nothing || CUDA.unsafe_free!(spectrum)
        succeeded || output === nothing || CUDA.unsafe_free!(output)
        if fft_plan !== nothing
            finalize(fft_plan.p)
            empty!(CUDA.CUFFT.idle_handles)
        end
    end
end

# Fold E into the coefficients of the independent real sky values.  This
# combines both helicities with the SAME scan weight; it never substitutes
# 2*real(<F+ exp(i*chi)>).  It needs no symmetry assumption about E or Mueller.
function _pack_spin_reduced_effective_kernel!(output, gemm, scan_weight::Int32,
        Mc::Int32, pixel_count::Int32, beam_count::Int32, batch_count::Int32)
    linear = (Int64(blockIdx().x)-1)*Int64(blockDim().x)+Int64(threadIdx().x)
    term_count = Int64(3)*batch_count
    total = term_count*beam_count*pixel_count
    linear > total && return
    index = linear-1
    term = index % term_count; index ÷= term_count
    beam = index % beam_count+1
    pixel = index÷beam_count+1
    tile = term÷(3Mc)
    t = term % (3Mc)
    G = (Mc-1)÷2
    if t < Mc
        M = Int32((t+1)÷2)
        spin, dual = Int32(1),Int32(1)
        imaginary = t > 0 && iseven(t)
    else
        p = t-Mc
        M = Int32(p÷2)-G
        spin, dual = Int32(2),Int32(3)
        imaginary = isodd(p)
    end
    batch = M+G+1+Mc*tile
    mirror = -M+G+1+Mc*tile
    @inbounds direct = gemm[spin+3*(beam-1),pixel,batch]
    if t == 0
        coefficient = direct
    else
        @inbounds reflected = gemm[dual+3*(beam-1),pixel,mirror]
        isodd(M) && (reflected = -reflected)
        if imaginary
            difference = direct-reflected
            coefficient = complex(-imag(difference),real(difference))
        else
            coefficient = direct+reflected
        end
    end
    @inbounds output[term+1,2beam-1,pixel,scan_weight] = real(coefficient)
    @inbounds output[term+1,2beam,pixel,scan_weight] = imag(coefficient)
    return
end

function _accumulate_spin_reduced_result_kernel!(accumulated, products,
        scan_weight::Int32, sky_count::Int32, beam_count::Int32,
        pixel_count::Int32)
    linear = (Int64(blockIdx().x)-1)*Int64(blockDim().x)+Int64(threadIdx().x)
    total = Int64(sky_count)*beam_count*pixel_count
    linear > total && return
    index = linear-1
    sky = index % sky_count+1; index ÷= sky_count
    beam = index % beam_count+1
    pixel = index÷beam_count+1
    @inbounds accumulated[sky,beam,pixel,scan_weight] +=
        complex(products[sky,2beam-1,pixel],products[sky,2beam,pixel])
    return
end

function accumulate_convolution_tile_gemm_gpu!(
        accumulated::CuArray{Complex{T},4}, sky::_SpinReducedSkyRing{T},
        effective::CuArray{T,4}) where {T<:_HARMONIC_PRECISIONS}
    CUDA.functional() || throw(ArgumentError("no functional GPU is available"))
    ns,terms,np = size(sky)
    nb = size(accumulated,2)
    terms == 3*(2sky.Mmax+1)*sky.l_count ||
        throw(DimensionMismatch("invalid reduced sky metadata"))
    size(effective) == (terms,2nb,np,2) ||
        throw(DimensionMismatch("spin-reduced effective-beam dimensions do not match sky"))
    size(accumulated) == (ns,nb,np,2) ||
        throw(DimensionMismatch("accumulator dimensions do not match sky"))
    all(x->0<x<=typemax(Int32),(ns,terms,np,2nb)) ||
        throw(ArgumentError("reduced GEMM dimensions must be positive and fit Int32"))
    products = CUDA.zeros(T,ns,2nb,np)
    try
        for w in 1:2
            _CONVOLUTION_CUBLAS.gemm_strided_batched!('N','N',one(T),
                sky.values,view(effective,:,:,:,w),zero(T),products)
            @cuda threads=_SKY_RING_THREADS blocks=cld(ns*nb*np,_SKY_RING_THREADS) _accumulate_spin_reduced_result_kernel!(
                accumulated,products,Int32(w),Int32(ns),Int32(nb),Int32(np))
        end
        CUDA.synchronize()
    finally
        CUDA.synchronize()
        CUDA.unsafe_free!(products)
    end
    return accumulated
end
