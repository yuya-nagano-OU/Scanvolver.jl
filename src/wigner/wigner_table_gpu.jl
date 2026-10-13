"""
Packed Chebyshev coefficients of `d^l_{Mn}(beta)`.

`coefficients[r, pair, l_index]` multiplies
`T_{r-1}(x(beta))`. Pair metadata contains only signed `(M, n)` pairs
satisfying `abs(M - n) <= kappa`.
"""
struct WignerChebyshevTable{T<:_CHEBYSHEV_PRECISIONS}
    lstart::Int
    lstop::Int
    kappa::Int
    nmax::Int
    beta_max::T
    pair_M::Vector{Int32}
    pair_n::Vector{Int32}
    first_l::Vector{Int32}
    coefficients::Array{T,3}
end

struct _WignerChebyshevGPUPlan{T<:_GPU_RECURSION_PRECISIONS}
    M::CuVector{Int32}
    n::CuVector{Int32}
    l0::CuVector{Int32}
    sign::CuVector{T}
    log_coefficient::CuVector{T}
    pow_c::CuVector{Int32}
    pow_s::CuVector{Int32}
end

struct _CuWignerChebyshevTable{T<:_GPU_RECURSION_PRECISIONS}
    coefficients::CuArray{T,3}
    pair_lookup::CuMatrix{Int32}
    lstart::Int32
    lstop::Int32
    kappa::Int32
    nmax::Int32
    Mmax::Int32
    beta_order::Int32
    beta_max::T
end

function _free_wigner_chebyshev_gpu_plan!(
    plan::_WignerChebyshevGPUPlan,
)
    CUDA.unsafe_free!(plan.M)
    CUDA.unsafe_free!(plan.n)
    CUDA.unsafe_free!(plan.l0)
    CUDA.unsafe_free!(plan.sign)
    CUDA.unsafe_free!(plan.log_coefficient)
    CUDA.unsafe_free!(plan.pow_c)
    CUDA.unsafe_free!(plan.pow_s)
    return nothing
end

function _check_wigner_chebyshev_inputs(
    cp,
    plan::ScanMomentPlan,
)
    cp.HWP == false || throw(ArgumentError(
        "Wigner-Chebyshev tables currently support HWP=false",
    ))
    cp.κ == plan.kappa || throw(ArgumentError(
        "cp.κ=$(cp.κ) does not match plan.kappa=$(plan.kappa)",
    ))
    nmax = calc_nmax(cp, cp.lstop)
    nmax == plan.nmax || throw(ArgumentError(
        "cp nmax=$nmax does not match plan.nmax=$(plan.nmax)",
    ))
    cp.lstop <= typemax(Int32) || throw(ArgumentError(
        "cp.lstop must fit in Int32 for packed table indexing",
    ))
    return nothing
end

function _build_wigner_chebyshev_gpu_plan(
    cp,
    scan_plan::ScanMomentPlan{T},
) where {T<:_GPU_RECURSION_PRECISIONS}
    pair_M = Int32[]
    pair_n = Int32[]
    first_l = Int32[]
    l0_values = Int32[]
    signs = T[]
    log_coefficients = T[]
    pow_cs = Int32[]
    pow_ss = Int32[]

    Mmax = calc_Mmax(cp, cp.lstop)
    for n in -scan_plan.nmax:scan_plan.nmax
        first_M = max(-Mmax, n - scan_plan.kappa)
        last_M = min(Mmax, n + scan_plan.kappa)
        for M in first_M:last_M
            l0 = max(abs(M), abs(n))
            l0 > cp.lstop && continue
            seed = _precompute_wignerd_lmin_seed_plan(l0, M, n)
            push!(pair_M, Int32(M))
            push!(pair_n, Int32(n))
            push!(first_l, Int32(max(cp.lstart, l0)))
            push!(l0_values, Int32(l0))
            push!(signs, T(seed.sign))
            push!(log_coefficients, T(seed.log_coefficient))
            push!(pow_cs, Int32(seed.pow_c))
            push!(pow_ss, Int32(seed.pow_s))
        end
    end
    isempty(pair_M) && throw(ArgumentError(
        "the Wigner-Chebyshev plan contains no valid (M, n) pairs",
    ))

    device_plan = _WignerChebyshevGPUPlan(
        CuArray(pair_M),
        CuArray(pair_n),
        CuArray(l0_values),
        CuArray(signs),
        CuArray(log_coefficients),
        CuArray(pow_cs),
        CuArray(pow_ss),
    )
    return pair_M, pair_n, first_l, device_plan
end

function _wigner_chebyshev_node_kernel!(
    node_values,
    pair_M,
    pair_n,
    pair_l0,
    seed_sign,
    seed_log_coefficient,
    seed_pow_c,
    seed_pow_s,
    beta::T,
    node_index::Int32,
    lstart::Int32,
    lstop::Int32,
    pair_count::Int32,
) where {T<:_GPU_RECURSION_PRECISIONS}
    pair_index =
        (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    pair_index > pair_count && return nothing

    @inbounds M = pair_M[pair_index]
    @inbounds n = pair_n[pair_index]
    @inbounds l0 = pair_l0[pair_index]
    cosine_beta = cos(beta)
    cosine_half = cos(T(0.5) * beta)
    sine_half = sin(T(0.5) * beta)
    previous_x = zero(T)
    previous_exponent = Int32(0)
    @inbounds current_x, current_exponent = _gpu_lmin_seed_scaled(
        seed_sign[pair_index],
        seed_log_coefficient[pair_index],
        seed_pow_c[pair_index],
        seed_pow_s[pair_index],
        cosine_half,
        sine_half,
    )

    if l0 >= lstart
        l_index = l0 - lstart + Int32(1)
        @inbounds node_values[node_index, pair_index, l_index] =
            _gpu_scaled_to_float(current_x, current_exponent)
    end

    for l in l0:(lstop - Int32(1))
        f0, f1, f2 = _gpu_l_step_coefficients(l, M, n, T)
        current_aligned, previous_aligned, exponent =
            _gpu_align_scaled_terms(
                current_x,
                current_exponent,
                previous_x,
                previous_exponent,
            )
        next_x =
            f0 * (cosine_beta - f1) * current_aligned -
            f2 * previous_aligned
        next_x, next_exponent =
            _gpu_normalize_scaled_pair(next_x, exponent)
        previous_x, previous_exponent =
            current_x, current_exponent
        current_x, current_exponent = next_x, next_exponent

        next_l = l + Int32(1)
        next_l < lstart && continue
        l_index = next_l - lstart + Int32(1)
        @inbounds node_values[node_index, pair_index, l_index] =
            _gpu_scaled_to_float(current_x, current_exponent)
    end
    return nothing
end

function _compute_wigner_chebyshev_nodes_gpu!(
    node_values,
    gpu_plan::_WignerChebyshevGPUPlan{T},
    cp,
    nodes::AbstractVector{T},
) where {T<:_GPU_RECURSION_PRECISIONS}
    size(node_values) == (
        length(nodes),
        length(gpu_plan.M),
        cp.lstop - cp.lstart + 1,
    ) || throw(DimensionMismatch(
        "node_values has incompatible dimensions",
    ))
    fill!(node_values, zero(T))
    pair_count = length(gpu_plan.M)
    blocks = cld(pair_count, _GPU_THREADS)
    for node_index in eachindex(nodes)
        @cuda threads=_GPU_THREADS blocks=blocks _wigner_chebyshev_node_kernel!(
            node_values,
            gpu_plan.M,
            gpu_plan.n,
            gpu_plan.l0,
            gpu_plan.sign,
            gpu_plan.log_coefficient,
            gpu_plan.pow_c,
            gpu_plan.pow_s,
            nodes[node_index],
            Int32(node_index),
            Int32(cp.lstart),
            Int32(cp.lstop),
            Int32(pair_count),
        )
    end
    return node_values
end

"""
    build_wigner_chebyshev_table_gpu(
        cp, plan; device_id=nothing,
    ) -> WignerChebyshevTable

Build the scan-independent Chebyshev coefficients of all required
`d^l_{Mn}(beta)` values. Only pairs satisfying `abs(M - n) <= cp.κ` are
evaluated. Wigner recurrence and the Chebyshev transform both run on the
selected GPU; the compact completed table is returned in host memory.
"""
function build_wigner_chebyshev_table_gpu(
    cp,
    scan_plan::ScanMomentPlan{T};
    device_id::Union{Nothing,Integer}=nothing,
) where {T<:_GPU_RECURSION_PRECISIONS}
    _check_wigner_chebyshev_inputs(cp, scan_plan)
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    device_id === nothing || CUDA.device!(Int(device_id))

    pair_M, pair_n, first_l, gpu_plan =
        _build_wigner_chebyshev_gpu_plan(cp, scan_plan)
    pair_count = length(pair_M)
    l_count = cp.lstop - cp.lstart + 1
    nodes = chebyshev_nodes(scan_plan)
    node_values = CUDA.zeros(
        T,
        scan_plan.beta_order,
        pair_count,
        l_count,
    )
    transform = nothing
    coefficient_matrix = nothing
    coefficients = nothing
    try
        _compute_wigner_chebyshev_nodes_gpu!(
            node_values,
            gpu_plan,
            cp,
            nodes,
        )

        transform = CuArray(_chebyshev_transform(scan_plan))
        coefficient_matrix = transform * reshape(
            node_values,
            scan_plan.beta_order,
            pair_count * l_count,
        )
        coefficients = Array(reshape(
            coefficient_matrix,
            scan_plan.beta_order,
            pair_count,
            l_count,
        ))
    finally
        CUDA.synchronize()
        coefficient_matrix === nothing ||
            CUDA.unsafe_free!(coefficient_matrix)
        transform === nothing || CUDA.unsafe_free!(transform)
        CUDA.unsafe_free!(node_values)
        _free_wigner_chebyshev_gpu_plan!(gpu_plan)
    end

    return WignerChebyshevTable{T}(
        cp.lstart,
        cp.lstop,
        scan_plan.kappa,
        scan_plan.nmax,
        scan_plan.beta_max,
        pair_M,
        pair_n,
        first_l,
        coefficients,
    )
end

function _check_wigner_chebyshev_table(table::WignerChebyshevTable)
    0 <= table.lstart <= table.lstop || throw(ArgumentError(
        "table bounds must satisfy 0 <= lstart <= lstop",
    ))
    table.lstop <= typemax(Int32) || throw(ArgumentError(
        "table.lstop must fit in Int32 for GPU indexing",
    ))
    table.kappa >= 0 || throw(ArgumentError(
        "table.kappa must be non-negative",
    ))
    table.nmax >= 0 || throw(ArgumentError(
        "table.nmax must be non-negative",
    ))
    isfinite(table.beta_max) && table.beta_max > 0 ||
        throw(ArgumentError(
            "table.beta_max must be finite and positive",
        ))
    pair_count = length(table.pair_M)
    pair_count > 0 || throw(ArgumentError(
        "table must contain at least one (M, n) pair",
    ))
    length(table.pair_n) == pair_count || throw(DimensionMismatch(
        "pair_n length does not match pair_M length",
    ))
    length(table.first_l) == pair_count || throw(DimensionMismatch(
        "first_l length does not match pair_M length",
    ))
    size(table.coefficients, 1) > 0 || throw(DimensionMismatch(
        "the Chebyshev coefficient axis must be non-empty",
    ))
    size(table.coefficients, 2) == pair_count ||
        throw(DimensionMismatch(
            "the coefficient pair axis does not match pair metadata",
        ))
    size(table.coefficients, 3) == table.lstop - table.lstart + 1 ||
        throw(DimensionMismatch(
            "the coefficient l axis does not match table bounds",
        ))
    @inbounds for pair_index in eachindex(table.pair_M)
        M = Int(table.pair_M[pair_index])
        n = Int(table.pair_n[pair_index])
        expected_first_l = max(table.lstart, abs(M), abs(n))
        table.first_l[pair_index] == expected_first_l ||
            throw(ArgumentError(
                "first_l does not match pair (M=$M, n=$n)",
            ))
        expected_first_l <= table.lstop || throw(ArgumentError(
            "pair (M=$M, n=$n) has no valid l in the table",
        ))
    end
    return nothing
end

"""
    upload_wigner_table_gpu(table; device_id=nothing)

Upload a host `WignerChebyshevTable` and construct its dense signed
`(M, n)` pair lookup on the selected GPU.
"""
function upload_wigner_table_gpu(
    table::WignerChebyshevTable{T};
    device_id::Union{Nothing,Integer}=nothing,
) where {T<:_GPU_RECURSION_PRECISIONS}
    _check_wigner_chebyshev_table(table)
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no functional GPU is available",
    ))
    device_id === nothing || CUDA.device!(Int(device_id))

    Mmax = maximum(abs, table.pair_M; init=Int32(0))
    pair_lookup = zeros(
        Int32,
        2 * Int(Mmax) + 1,
        2 * table.nmax + 1,
    )
    @inbounds for pair_index in eachindex(table.pair_M)
        M = Int(table.pair_M[pair_index])
        n = Int(table.pair_n[pair_index])
        abs(n) <= table.nmax || throw(ArgumentError(
            "pair n=$n is outside table.nmax=$(table.nmax)",
        ))
        abs(M - n) <= table.kappa || throw(ArgumentError(
            "pair (M=$M, n=$n) is outside the kappa band",
        ))
        row = M + Int(Mmax) + 1
        column = n + table.nmax + 1
        pair_lookup[row, column] == 0 || throw(ArgumentError(
            "table contains duplicate pair (M=$M, n=$n)",
        ))
        pair_lookup[row, column] = Int32(pair_index)
    end

    return _CuWignerChebyshevTable{T}(
        CuArray(table.coefficients),
        CuArray(pair_lookup),
        Int32(table.lstart),
        Int32(table.lstop),
        Int32(table.kappa),
        Int32(table.nmax),
        Mmax,
        Int32(size(table.coefficients, 1)),
        table.beta_max,
    )
end
