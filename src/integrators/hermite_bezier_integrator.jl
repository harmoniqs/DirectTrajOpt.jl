export HermiteBezierIntegrator
export HermiteBezierDerivativeIntegrator

# TODO: Why don't we have sparsity selectors for this and derivative?

# -------------------------------------------------------------- #
# Midpoints
# -------------------------------------------------------------- #

"""
    HermiteBezierIntegrator(u_name, du_name, u_b1_name, u_b2_name, traj)

Interval equality constraint for cubic Hermite control-value Bézier points.

For each interval k = 1:N-1, enforces

    3 * (u_b1[k] - u[k])       - Δt[k] * du[k]   = 0
    3 * (u[k+1] - u_b2[k])     - Δt[k] * du[k+1] = 0

Thus the cubic Hermite segment has Bézier control points

    u[k], u_b1[k], u_b2[k], u[k+1].

If u, u_b1, and u_b2 are bounded, then the full cubic Hermite control
value is bounded over the entire interval by the Bézier convex-hull property.
"""
struct HermiteBezierIntegrator{F} <: AbstractIntegrator
    f::F

    # Keep x_name for DirectTrajOpt's generic integrator infrastructure.
    x_name::Symbol

    u_name::Symbol
    du_name::Symbol
    u_b1_name::Symbol
    u_b2_name::Symbol

    x_dim::Int
    var_dim::Int
    dim::Int

    function HermiteBezierIntegrator(
        u_name::Symbol,
        du_name::Symbol,
        u_b1_name::Symbol,
        u_b2_name::Symbol,
        traj::NamedTrajectory,
    )
        @assert u_name ∈ traj.names "Control variable $u_name not found in trajectory"
        @assert du_name ∈ traj.names "Derivative variable $du_name not found in trajectory"
        @assert u_b1_name ∈ traj.names "Bézier point variable $u_b1_name not found in trajectory"
        @assert u_b2_name ∈ traj.names "Bézier point variable $u_b2_name not found in trajectory"
        @assert traj.timestep isa Symbol "HermiteBezierIntegrator requires a symbolic timestep variable"
        @assert traj.timestep ∈ traj.names "Timestep variable $(traj.timestep) not found in trajectory"

        x_dim = traj.dims[u_name]

        @assert traj.dims[du_name] == x_dim (
            "Dimension mismatch: $du_name has dim $(traj.dims[du_name]), " *
            "but $u_name has dim $x_dim"
        )

        @assert traj.dims[u_b1_name] == x_dim (
            "Dimension mismatch: $u_b1_name has dim $(traj.dims[u_b1_name]), " *
            "but $u_name has dim $x_dim"
        )

        @assert traj.dims[u_b2_name] == x_dim (
            "Dimension mismatch: $u_b2_name has dim $(traj.dims[u_b2_name]), " *
            "but $u_name has dim $x_dim"
        )

        u_dim = traj.dims[u_name]

        # Each interval contributes two u_dim-sized Bézier equations:
        #   3(u_b1 - u) - Δt * du = 0
        #   3(u[k+1] - u_b2) - Δt * du[k+1] = 0
        # So the per-interval residual block is 2*u_dim wide.
        x_dim = 2 * u_dim
        dim = x_dim * (traj.N - 1)

        # Depends on two adjacent full knot vectors:
        # z[k] contains u[k], du[k], u_b1[k], u_b2[k], Δt[k]
        # z[k+1] contains u[k+1], du[k+1]
        var_dim = 6 * u_dim + 1

        f = (uₖ₊₁, uₖ, duₖ, duₖ₊₁, u_b1ₖ, u_b2ₖ, Δtₖ) -> [
            3 .* (u_b1ₖ .- uₖ) .- Δtₖ .* duₖ
            3 .* (uₖ₊₁ .- u_b2ₖ) .- Δtₖ .* duₖ₊₁
        ]

        return new{typeof(f)}(
            f,
            u_name,
            u_name,
            du_name,
            u_b1_name,
            u_b2_name,
            x_dim,
            var_dim,
            dim,
        )
    end
end

function Base.show(io::IO, H::HermiteBezierIntegrator)
    print(
        io,
        "HermiteBezierIntegrator: " *
        ":$(H.u_b1_name), :$(H.u_b2_name) define cubic Bézier points for :$(H.u_name) " *
        "(dim = $(H.x_dim))",
    )
end

function evaluate!(
    δ::AbstractVector,
    H::HermiteBezierIntegrator,
    traj::NamedTrajectory,
)
    for k = 1:(traj.N - 1)
        uₖ = traj[k][H.u_name]
        uₖ₊₁ = traj[k + 1][H.u_name]

        duₖ = traj[k][H.du_name]
        duₖ₊₁ = traj[k + 1][H.du_name]

        u_b1ₖ = traj[k][H.u_b1_name]
        u_b2ₖ = traj[k][H.u_b2_name]

        Δtₖ = traj[k].timestep

        δ[slice(k, H.x_dim)] =
            H.f(uₖ₊₁, uₖ, duₖ, duₖ₊₁, u_b1ₖ, u_b2ₖ, Δtₖ)
    end

    return nothing
end

@views function eval_jacobian(
    H::HermiteBezierIntegrator,
    traj::NamedTrajectory,
)
    ∂H = spzeros(H.dim, traj.dim * traj.N + traj.global_dim)

    u_dim = traj.dims[H.u_name]
    u_idx = traj.components[H.u_name]
    du_idx = traj.components[H.du_name]
    u_b1_idx = traj.components[H.u_b1_name]
    u_b2_idx = traj.components[H.u_b2_name]
    Δt_idx = traj.components[traj.timestep]

    for k = 1:(traj.N - 1)
        Δtₖ = traj[k].timestep
        duₖ = traj[k][H.du_name]
        duₖ₊₁ = traj[k + 1][H.du_name]

        r₁ = slice(k, 1:u_dim, H.x_dim)
        r₂ = slice(k, (u_dim + 1):(2 * u_dim), H.x_dim)

        u_cols = slice(k, u_idx, traj.dim)
        du_cols = slice(k, du_idx, traj.dim)
        u_b1_cols = slice(k, u_b1_idx, traj.dim)
        u_b2_cols = slice(k, u_b2_idx, traj.dim)
        Δt_col = first(slice(k, Δt_idx, traj.dim))

        uₖ₊₁_cols = slice(k + 1, u_idx, traj.dim)
        duₖ₊₁_cols = slice(k + 1, du_idx, traj.dim)

        for i = 1:u_dim
            # 3 * (u_b1[k] - u[k]) - Δt[k] * du[k]
            ∂H[r₁[i], u_cols[i]] = -3
            ∂H[r₁[i], u_b1_cols[i]] = 3
            ∂H[r₁[i], du_cols[i]] = -Δtₖ
            ∂H[r₁[i], Δt_col] = -duₖ[i]

            # 3 * (u[k+1] - u_b2[k]) - Δt[k] * du[k+1]
            ∂H[r₂[i], uₖ₊₁_cols[i]] = 3
            ∂H[r₂[i], u_b2_cols[i]] = -3
            ∂H[r₂[i], duₖ₊₁_cols[i]] = -Δtₖ
            ∂H[r₂[i], Δt_col] = -duₖ₊₁[i]
        end
    end

    return ∂H
end

# @views function eval_hessian_of_lagrangian(
#     H::HermiteBezierIntegrator,
#     traj::NamedTrajectory,
#     μ::AbstractVector,
# )
#     return spzeros(
#         traj.dim * traj.N + traj.global_dim,
#         traj.dim * traj.N + traj.global_dim,
#     )
# end


# TODO: fix
@views function eval_hessian_of_lagrangian(
    H::HermiteBezierIntegrator,
    traj::NamedTrajectory,
    μ::AbstractVector,
)
    μ∂²H = spzeros(
        traj.dim * traj.N + traj.global_dim,
        traj.dim * traj.N + traj.global_dim,
    )

    u_dim = traj.dims[H.u_name]
    du_idx = traj.components[H.du_name]
    Δt_idx = traj.components[traj.timestep]

    for k = 1:(traj.N - 1)
        μₖ = μ[slice(k, H.x_dim)]

        du_cols = slice(k, du_idx, traj.dim)
        duₖ₊₁_cols = slice(k + 1, du_idx, traj.dim)
        Δt_col = first(slice(k, Δt_idx, traj.dim))

        for i = 1:u_dim
            # ∂²/∂Δt∂du of μᵢ * (-Δt * duᵢ) is -μᵢ.
            μ∂²H[Δt_col, du_cols[i]] += -μₖ[i]
            μ∂²H[du_cols[i], Δt_col] += -μₖ[i]

            # Same bilinear term for the second Bézier equation, with du[k+1].
            μ∂²H[Δt_col, duₖ₊₁_cols[i]] += -μₖ[u_dim + i]
            μ∂²H[duₖ₊₁_cols[i], Δt_col] += -μₖ[u_dim + i]
        end
    end

    return μ∂²H
end

# -------------------------------------------------------------- #
# Derivative midpoint
# -------------------------------------------------------------- #

"""
    HermiteDerivativeMidpointIntegrator(u_name, du_name, du_mid_name, traj)

Interval equality constraint for cubic Hermite controls.

For each interval k = 1:N-1, enforces

    3 * (u[k+1] - u[k]) - Δt[k] * (du[k] + du[k+1] + du_mid[k]) = 0

equivalently,

    du_mid[k] = 3 * (u[k+1] - u[k]) / Δt[k] - du[k] - du[k+1]

when Δt[k] != 0.

For a cubic Hermite segment, the derivative is a quadratic Bézier curve
with control points

    du[k], du_mid[k], du[k+1].

Therefore, bounding both `du` and `du_mid` bounds the derivative over the
entire interval by the Bézier convex-hull property.
"""
struct HermiteBezierDerivativeIntegrator{F} <: AbstractIntegrator
    f::F

    # Keep x_name because DirectTrajOpt.Solvers.get_nonlinear_constraints
    # currently expects each integrator to expose x_name, x_names, or t_name.
    x_name::Symbol

    u_name::Symbol
    du_name::Symbol
    du_mid_name::Symbol

    x_dim::Int
    var_dim::Int
    dim::Int

    function HermiteBezierDerivativeIntegrator(
        u_name::Symbol,
        du_name::Symbol,
        du_mid_name::Symbol,
        traj::NamedTrajectory,
    )
        @assert u_name ∈ traj.names "Control variable $u_name not found in trajectory"
        @assert du_name ∈ traj.names "Derivative variable $du_name not found in trajectory"
        @assert du_mid_name ∈ traj.names "Middle derivative control-point variable $du_mid_name not found in trajectory"
        @assert traj.timestep isa Symbol "HermiteDerivativeMidpointIntegrator requires a symbolic timestep variable"
        @assert traj.timestep ∈ traj.names "Timestep variable $(traj.timestep) not found in trajectory"

        x_dim = traj.dims[u_name]

        @assert traj.dims[du_name] == x_dim (
            "Dimension mismatch: $du_name has dim $(traj.dims[du_name]), " *
            "but $u_name has dim $x_dim"
        )

        @assert traj.dims[du_mid_name] == x_dim (
            "Dimension mismatch: $du_mid_name has dim $(traj.dims[du_mid_name]), " *
            "but $u_name has dim $x_dim"
        )

        # Variables used per interval:
        #   u[k], du[k], du_mid[k], Δt[k], u[k+1], du[k+1]
        #
        # The generic integrator sparsity structure covers two adjacent full
        # knot vectors, which is exactly what this depends on.
        var_dim = 5 * x_dim + 1

        # One x_dim residual per interval.
        dim = x_dim * (traj.N - 1)

        f = (uₖ₊₁, uₖ, duₖ, duₖ₊₁, du_midₖ, Δtₖ) ->
            3 .* (uₖ₊₁ .- uₖ) .- Δtₖ .* (duₖ .+ duₖ₊₁ .+ du_midₖ)

        return new{typeof(f)}(
            f,
            u_name,
            u_name,
            du_name,
            du_mid_name,
            x_dim,
            var_dim,
            dim,
        )
    end
end

function Base.show(io::IO, D::HermiteBezierDerivativeIntegrator)
    print(
        io,
        "HermiteDerivativeMidpointIntegrator: " *
        "3(:$(D.u_name)[k+1] - :$(D.u_name)[k]) = " *
        "Δt * (:$(D.du_name)[k] + :$(D.du_name)[k+1] + :$(D.du_mid_name)[k]) " *
        "(dim = $(D.x_dim))",
    )
end

function evaluate!(
    δ::AbstractVector,
    D::HermiteBezierDerivativeIntegrator,
    traj::NamedTrajectory,
)
    for k = 1:(traj.N - 1)
        uₖ = traj[k][D.u_name]
        uₖ₊₁ = traj[k + 1][D.u_name]

        duₖ = traj[k][D.du_name]
        duₖ₊₁ = traj[k + 1][D.du_name]

        du_midₖ = traj[k][D.du_mid_name]
        Δtₖ = traj[k].timestep

        δ[slice(k, D.x_dim)] =
            D.f(uₖ₊₁, uₖ, duₖ, duₖ₊₁, du_midₖ, Δtₖ)
    end

    return nothing
end

@views function eval_jacobian(
    D::HermiteBezierDerivativeIntegrator,
    traj::NamedTrajectory,
)
    ∂D = spzeros(D.dim, traj.dim * traj.N + traj.global_dim)

    u_dim = traj.dims[D.u_name]
    u_idx = traj.components[D.u_name]
    du_idx = traj.components[D.du_name]
    du_mid_idx = traj.components[D.du_mid_name]
    Δt_idx = traj.components[traj.timestep]

    for k = 1:(traj.N - 1)
        Δtₖ = traj[k].timestep
        duₖ = traj[k][D.du_name]
        duₖ₊₁ = traj[k + 1][D.du_name]
        du_midₖ = traj[k][D.du_mid_name]

        rows = slice(k, D.x_dim)

        u_cols = slice(k, u_idx, traj.dim)
        du_cols = slice(k, du_idx, traj.dim)
        du_mid_cols = slice(k, du_mid_idx, traj.dim)
        Δt_col = first(slice(k, Δt_idx, traj.dim))

        uₖ₊₁_cols = slice(k + 1, u_idx, traj.dim)
        duₖ₊₁_cols = slice(k + 1, du_idx, traj.dim)

        for i = 1:u_dim
            # 3 * (u[k+1] - u[k]) - Δt[k] * (du[k] + du[k+1] + du_mid[k])
            ∂D[rows[i], u_cols[i]] = -3
            ∂D[rows[i], uₖ₊₁_cols[i]] = 3
            ∂D[rows[i], du_cols[i]] = -Δtₖ
            ∂D[rows[i], duₖ₊₁_cols[i]] = -Δtₖ
            ∂D[rows[i], du_mid_cols[i]] = -Δtₖ
            ∂D[rows[i], Δt_col] = -(duₖ[i] + duₖ₊₁[i] + du_midₖ[i])
        end
    end

    return ∂D
end

@views function eval_hessian_of_lagrangian(
    D::HermiteBezierDerivativeIntegrator,
    traj::NamedTrajectory,
    μ::AbstractVector,
)
    μ∂²D = spzeros(
        traj.dim * traj.N + traj.global_dim,
        traj.dim * traj.N + traj.global_dim,
    )

    u_dim = traj.dims[D.u_name]
    du_idx = traj.components[D.du_name]
    du_mid_idx = traj.components[D.du_mid_name]
    Δt_idx = traj.components[traj.timestep]

    for k = 1:(traj.N - 1)
        μₖ = μ[slice(k, D.x_dim)]

        du_cols = slice(k, du_idx, traj.dim)
        du_mid_cols = slice(k, du_mid_idx, traj.dim)
        duₖ₊₁_cols = slice(k + 1, du_idx, traj.dim)
        Δt_col = first(slice(k, Δt_idx, traj.dim))

        for i = 1:u_dim
            # ∂²/∂Δt∂du of μᵢ * (-Δt * (duᵢ + du_nextᵢ + du_midᵢ)) is -μᵢ.
            for col in (du_cols[i], duₖ₊₁_cols[i], du_mid_cols[i])
                μ∂²D[Δt_col, col] += -μₖ[i]
                μ∂²D[col, Δt_col] += -μₖ[i]
            end
        end
    end

    return μ∂²D
end

# =========================================================================== #

@testitem "testing HermiteBezierIntegrator" begin
    include("../../test/test_utils.jl")

    traj = named_trajectory_type_1()

    # This test trajectory is expected to have :a and :da.
    a_b1_data = copy(traj[:a])
    a_b2_data = copy(traj[:a])

    traj = add_component(
        traj,
        :a_b1,
        a_b1_data;
        type = :control,
    )

    traj = add_component(
        traj,
        :a_b2,
        a_b2_data;
        type = :control,
    )

    H = HermiteBezierIntegrator(:a, :da, :a_b1, :a_b2, traj)

    @test H.x_dim == 2 * traj.dims[:a]
    @test H.dim == H.x_dim * (traj.N - 1)
    @test H.var_dim == 6 * traj.dims[:a] + 1

    test_integrator(H, traj)
end

@testitem "testing HermiteDerivativeMidpointIntegrator" begin
    include("../../test/test_utils.jl")

    traj = named_trajectory_type_1()

    # This test trajectory is expected to have :a and :da.
    # Add a middle derivative control-point variable with the same dimension.
    da_mid_data = zeros(size(traj[:da]))

    traj = add_component(
        traj,
        :da_mid,
        da_mid_data;
        type = :control,
    )

    D = HermiteBezierDerivativeIntegrator(:a, :da, :da_mid, traj)

    test_integrator(D, traj)
end
