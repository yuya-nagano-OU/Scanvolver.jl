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
