# Minimal rotation helpers required by the Angles constructor.

Rz(α) = @SMatrix [
    cos(α) -sin(α) 0.0
    sin(α)  cos(α) 0.0
    0.0     0.0    1.0
]

Ry(β) = @SMatrix [
     cos(β) 0.0 sin(β)
     0.0    1.0 0.0
    -sin(β) 0.0 cos(β)
]

Rzyz(α, β, γ) = Rz(α) * Ry(β) * Rz(γ)

wrap_pm_pi(x) = mod(x + π, 2π) - π

"""
    euler_zyz_from_R(R; eps=1e-12)

ZYZ Euler angles of a rotation matrix, with `alpha` and `gamma` wrapped to
`(-pi, pi]`.

`beta` is taken from `atan(sy, R[3,3])` with `sy = sqrt(R[1,3]^2 + R[2,3]^2)`,
not from `acos(R[3,3])`, which loses half of its significant digits as `R[3,3]`
approaches 1. `hypot` is unnecessary here because every entry of a rotation
matrix is bounded by 1, and `sqrt` is much cheaper.

`eps` guards the degenerate cases `beta = 0` and `beta = pi`, where `alpha` and
`gamma` are individually undefined and only their sum (or difference) is
determined. **The test is on `sy`, never on `beta`.** `sy` is exactly the
quantity whose roundoff makes `atan(R[2,3], R[1,3])` meaningless, so it is the
only meaningful thing to threshold. Testing `sin(beta)` instead cannot work:
`acos` turns a one-ulp error in `R[3,3]` into `beta` of about 1.5e-8, far above
any sensible threshold, so a residual rotation that is numerically the identity
still takes the general branch and returns an arbitrary `alpha` and `gamma`.
Their sum then no longer reproduces the input rotation, which shifts every
`n != 0` beam mode by an O(1) phase and silently corrupts polarization.
"""
function euler_zyz_from_R(
    R::AbstractMatrix{<:Real};
    eps::Real=1e-12,
)
    sy = sqrt(R[1, 3] * R[1, 3] + R[2, 3] * R[2, 3])
    β = atan(sy, R[3, 3])

    if sy > eps
        α = atan(R[2, 3], R[1, 3])
        γ = atan(R[3, 2], -R[3, 1])
        return wrap_pm_pi(α), β, wrap_pm_pi(γ)
    elseif R[3, 3] > 0
        γ = atan(R[2, 1], R[1, 1])
        return 0.0, 0.0, wrap_pm_pi(γ)
    else
        # At beta = pi only alpha - gamma is determined. The upper-left block is
        # [-cos(d) -sin(d); -sin(d) cos(d)] with d = alpha - gamma, so
        # atan(-R[2,1], -R[1,1]) is +d and the gamma that goes with alpha = 0 is
        # -d, not +d.
        γ = atan(R[2, 1], -R[1, 1])
        return 0.0, π, wrap_pm_pi(γ)
    end
end

function split_rotation(
    phi,
    theta,
    dphi,
    dtheta,
    psi;
    eps=1e-12,
)
    center = Rzyz(phi, theta, 0.0)
    target = Rzyz(phi + dphi, theta + dtheta, psi)
    residual = transpose(center) * target
    return euler_zyz_from_R(Matrix(residual); eps=eps)
end

function split_rotation_vectors(
    phi,
    theta,
    dphi,
    dtheta,
    psi;
    eps=1e-12,
)
    sample_count = length(phi)
    length(theta) == sample_count ||
        throw(ArgumentError("theta must match phi length"))
    length(dphi) == sample_count ||
        throw(ArgumentError("dphi must match phi length"))
    length(dtheta) == sample_count ||
        throw(ArgumentError("dtheta must match phi length"))
    length(psi) == sample_count ||
        throw(ArgumentError("psi must match phi length"))

    α = similar(phi, Float64)
    β = similar(phi, Float64)
    γ = similar(phi, Float64)

    @inbounds for index in eachindex(phi, theta, dphi, dtheta, psi)
        α[index], β[index], γ[index] = split_rotation(
            phi[index],
            theta[index],
            dphi[index],
            dtheta[index],
            psi[index];
            eps=eps,
        )
    end

    return α, β, γ
end

function split_rotation_vectors(
    phi::Real,
    theta::Real,
    dphi::AbstractVector,
    dtheta::AbstractVector,
    psi::AbstractVector;
    eps=1e-12,
)
    sample_count = length(dphi)
    length(dtheta) == sample_count ||
        throw(ArgumentError("dtheta must match dphi length"))
    length(psi) == sample_count ||
        throw(ArgumentError("psi must match dphi length"))

    α = similar(dphi, Float64)
    β = similar(dphi, Float64)
    γ = similar(dphi, Float64)

    @inbounds for index in eachindex(dphi, dtheta, psi)
        α[index], β[index], γ[index] = split_rotation(
            phi,
            theta,
            dphi[index],
            dtheta[index],
            psi[index];
            eps=eps,
        )
    end

    return α, β, γ
end
