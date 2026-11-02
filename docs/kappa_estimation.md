# Estimating the kappa band

`estimate_kappa` is a CPU-only candidate selector for the signed band
`abs(M-n) <= kappa`. It does not change a convolution configuration or run
GPU work, and does not require a sky or a beam file.

```julia
using Scanvolver

kappa = estimate_kappa(; nside=512, lmax=1000, nmax=200)  # 9, default ε=1e-5
cp = ConvolutionParams(
    nside_output=512, lmax=1000, lstop=1000, bnmax=200, κ=kappa,
)
```

Output `nside`, `lmax`, and beam `nmax` are required **keyword-only** inputs.
The positive **amplitude threshold** `ε` defaults to `1e-5` and is the only
input that can be supplied positionally. Explicit keyword `ε` is also accepted:

```julia
estimate_kappa(1e-8; nside=512, lmax=1000, nmax=200)  # 12
estimate_kappa(; nside=512, lmax=1000, nmax=200, ε=1e-8)  # 12
```

Do not supply `ε` both ways in one call; that raises `ArgumentError`. Julia
allows geometry keywords in any order. Omitting one raises `UndefKeywordError`.
Multi-positional geometry calls such as `estimate_kappa(512, 1000, 200, 1e-8)`
are still rejected with `MethodError`.

By default the criterion is

```math
R(q)=\max_{|n|\le n_{\max},\ |M-n|=q}
\frac{|d^{\ell_{\max}}_{Mn}(\beta_{\max})|}
     {|d^{\ell_{\max}}_{nn}(\beta_{\max})|}.
```

The returned integer is the smallest non-negative kappa for which all
omitted `q > kappa` have `R(q) <= ε`. The center is the diagonal
at the **same angle**, not the value at zero angle or the matrix peak.
In particular, near-diagonal entries can be larger than the center.

## Evaluation angle and diagnostics

The default angle is `automatic_beta_max(nside; precision=Float64)`: the
HEALPix maximum center-to-corner angle with the existing numerical allowance.
To use the exact bound from another scan plan, pass it explicitly:

```julia
beta = automatic_beta_max(512; precision=Float32)
info = estimate_kappa(1e-8; nside=512, lmax=1000, nmax=200,
                      beta_max=beta, details=true)
info.kappa
info.beta_max
info.q_values, info.ratios
info.minimum_diagonal
info.tail_start, info.tail_bound
info.omitted_maximum_bound
info.threshold  # The requested ε, retained under the existing diagnostics name.
```

`ratios` contains the evaluated profile at `q_values`, starting with q=0.
Entries from `tail_start` onward are **not** filled with fictitious zero values:
they are excluded by `tail_bound`. The magnitude is unchanged under
simultaneous sign reversal of `(M,n)`, so non-negative n and both directions
in M cover the full signed domain without materializing a dense matrix.

## Numerical method and scope

The utility reuses the scaled Float64 CPU Wigner recurrence. It does not
truncate at the first sub-threshold element, since a profile need not be
monotone. To avoid evaluating every far-away component, it uses the bound

```math
|d^\ell_{Mn}(\beta)|\le
\sum_{r=q}^{\infty}\frac{x^r}{r!}
\le e^x\frac{x^q}{q!},\qquad x=\ell|\beta|,\quad q=|M-n|.
```

This follows from `d=exp(-i beta J_y)`, the tridiagonal generator, and
`norm(beta J_y, 2) = ell*abs(beta)`. Once `q > x`, the bound decreases with
q and bounds every still more distant component. Diagonal normalization
divides this bound by the smallest evaluated diagonal. Float64 recurrence
roundoff is not a certified part of this analytical bound.

**This evaluates only `ell=lmax` at the stated angle.** It is not a rigorous
supremum over every lower ell or every angle between zero and the bound.
An interior diagonal zero can make the ratio singular even if the endpoint
is well-behaved. The returned kappa is a starting point for convergence tests,
not a guarantee about IQU map error or an optimal beam-weighted truncation.
No FWHM-dependent high-ell suppression is assumed.

If any endpoint diagonal is at or below `diagonal_atol` (default
`sqrt(eps(Float64))`), diagonal normalization throws `DomainError`. To test
absolute Wigner amplitudes instead, explicitly choose another criterion:

```julia
kappa = estimate_kappa(; nside=512, lmax=1000, nmax=200, ε=1e-2,
                       normalization=:absolute)  # 4
```

This is a different threshold definition, never an automatic fallback.
