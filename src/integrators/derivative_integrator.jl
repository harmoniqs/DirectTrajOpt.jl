export DerivativeIntegrator

"""
    DerivativeIntegrator <: AbstractIntegrator

Integrator for derivative constraints of the form xₖ₊₁ - xₖ - Δt * ẋₖ = 0.

This enforces smoothness by relating a variable to its derivative.

# Fields
- `f::Function`: Constraint function f(xₖ₊₁, xₖ, ẋₖ, Δtₖ) = xₖ₊₁ - xₖ - Δtₖ * ẋₖ
- `x_name::Symbol`: Variable name
- `ẋ_name::Symbol`: Derivative variable name
- `x_dim::Int`: Dimension of variable
- `var_dim::Int`: Combined dimension (2*x_dim + 1 for xₖ, ẋₖ, Δtₖ, xₖ₊₁)
- `dim::Int`: Total constraint dimension (x_dim * (N-1))
- `∂fs::Vector{SparseMatrixCSC{Float64, Int}}`: Compact Jacobian storage
- `μ∂²fs::Vector{SparseMatrixCSC{Float64, Int}}`: Compact Hessian storage

# Example
```julia
# Enforce velocity smoothness: vₖ₊₁ - vₖ - Δt * aₖ = 0
integrator = DerivativeIntegrator(:v, :a, traj)
```
"""
struct DerivativeIntegrator{F} <: AbstractIntegrator
    f::F
    x_name::Symbol
    ẋ_name::Symbol
    x_dim::Int
    var_dim::Int
    dim::Int

    function DerivativeIntegrator(x::Symbol, ẋ::Symbol, traj::NamedTrajectory)
        x_dim = traj.dims[x]
        N = traj.N

        # Variables: [xₖ, ẋₖ, Δtₖ, xₖ₊₁]
        var_dim = 2*x_dim + 1 + x_dim  # = 3*x_dim + 1

        # Total constraint dimension
        dim = x_dim * (N - 1)

        # Define f function: constraint is f(xₖ₊₁, xₖ, ẋₖ, Δtₖ) = 0
        f = (xₖ₊₁, xₖ, ẋₖ, Δtₖ) -> xₖ₊₁ - xₖ - Δtₖ * ẋₖ

        return new{typeof(f)}(f, x, ẋ, x_dim, var_dim, dim)
    end
end

function Base.show(io::IO, D::DerivativeIntegrator)
    print(io, "DerivativeIntegrator: :$(D.x_name) += Δt * :$(D.ẋ_name)  (dim = $(D.x_dim))")
end

function evaluate!(δ::AbstractVector, D::DerivativeIntegrator, traj::NamedTrajectory)
    for k = 1:(traj.N-1)
        xₖ = traj[k][D.x_name]
        xₖ₊₁ = traj[k+1][D.x_name]
        ẋₖ = traj[k][D.ẋ_name]
        Δtₖ = traj[k].timestep
        δ[slice(k, D.x_dim)] = D.f(xₖ₊₁, xₖ, ẋₖ, Δtₖ)
    end
    return nothing
end

# Jacobian methods

@views function eval_jacobian(D::DerivativeIntegrator, traj::NamedTrajectory)
    if traj.warp !== nothing
        return _eval_jacobian_warped(D, traj)
    end
    ∂D = spzeros(D.dim, traj.dim * traj.N + traj.global_dim)
    for k = 1:(traj.N-1)
        ForwardDiff.jacobian!(
            ∂D[slice(k, D.x_dim), slice(k, 1:2traj.dim, traj.dim)],
            zz -> begin
                zₖ = zz[1:traj.dim]
                zₖ₊₁ = zz[(traj.dim+1):end]
                xₖ = zₖ[traj.components[D.x_name]]
                ẋₖ = zₖ[traj.components[D.ẋ_name]]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]
                xₖ₊₁ = zₖ₊₁[traj.components[D.x_name]]
                return D.f(xₖ₊₁, xₖ, ẋₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )
    end
    return ∂D
end

"""
    _eval_jacobian_warped(D, traj)

Phase 1b (DTO#149): the chain rows under a time warp. The defect is
`xₖ₊₁ − xₖ − Δtₖ(T)·ẋₖ` with Δtₖ DERIVED — the SmoothPulse chain survives the
BilinearIntegrator demotion arc (it is pulse-parameterization math, integrator-agnostic),
so the Jacobian is the packed two-knot window plus the EXACT warp column:
`∂δₖ/∂θⱼ = ∂Φ/∂Δtₖ · ∂Δtₖ/∂θⱼ` with `Φ = −Δtₖ·ẋₖ`, computed by ForwardDiff through
`with_params` (Dual-generic — exact for `GlobalScale`, whose `ddeltats_dparams`
override supplies the rational lattice weights).
"""
@views function _eval_jacobian_warped(D::DerivativeIntegrator, traj::NamedTrajectory)
    warp = traj.warp::AbstractTimeWarp   # the _warped path is reached only under a warp
    nθ = n_params(warp)
    θ0 = warp_params(warp)
    zdim = WarpPlumbing.packed_knot_dim(traj)
    ∂D = spzeros(D.dim, WarpPlumbing.packed_length(traj))

    # non-derived datavec rows per knot, and their indices within the pruned window
    dt_first = first(traj.components[traj.timestep])
    dt_dim = traj.dims[traj.timestep]
    rows = [r for r = 1:traj.dim if !(dt_first ≤ r < dt_first + dt_dim)]
    loc(r) = r < dt_first ? r : r - dt_dim
    x_loc = [loc(r) for r in traj.components[D.x_name]]
    ẋ_loc = [loc(r) for r in traj.components[D.ẋ_name]]

    for k = 1:(traj.N-1)
        cols = vcat(
            WarpPlumbing.packed_two_knot_window(traj, k),
            WarpPlumbing.warp_param_indices(traj),
        )
        ForwardDiff.jacobian!(
            ∂D[slice(k, D.x_dim), cols],
            zz -> begin
                w = zz[1:(end-nθ)]
                θ = zz[(end-nθ+1):end]
                Δtₖ = deltats(with_params(warp, θ), traj.N)[k]
                xₖ = w[x_loc]
                ẋₖ = w[ẋ_loc]
                xₖ₊₁ = w[zdim .+ x_loc]
                return D.f(xₖ₊₁, xₖ, ẋₖ, Δtₖ)
            end,
            [traj[k].data[rows]; traj[k+1].data[rows]; θ0],
        )
    end
    return ∂D
end

# Hessian methods

function eval_hessian_of_lagrangian(
    D::DerivativeIntegrator,
    traj::NamedTrajectory,
    μ::AbstractVector,
)
    if traj.warp !== nothing
        return _eval_hessian_of_lagrangian_warped(D, traj, μ)
    end
    μ∂²D = spzeros(traj.dim * traj.N + traj.global_dim, traj.dim * traj.N + traj.global_dim)

    for k = 1:(traj.N-1)
        μₖ = μ[slice(k, D.x_dim)]

        μ∂²Dₖ = ForwardDiff.hessian(
            zz -> begin
                zₖ = zz[1:traj.dim]
                zₖ₊₁ = zz[(traj.dim+1):end]
                xₖ = zₖ[traj.components[D.x_name]]
                ẋₖ = zₖ[traj.components[D.ẋ_name]]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]
                xₖ₊₁ = zₖ₊₁[traj.components[D.x_name]]
                return μₖ'D.f(xₖ₊₁, xₖ, ẋₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )

        μ∂²D[slice(k, 1:2traj.dim, traj.dim), slice(k, 1:2traj.dim, traj.dim)] .+= μ∂²Dₖ
    end
    return μ∂²D
end

"""
    _eval_hessian_of_lagrangian_warped(D, traj, μ)

Warp-path Hessian: ForwardDiff over `[pruned two-knot window; warp params]` of
`μₖ'f`, scattered into packed coordinates — the (window × warp) cross terms
(`∂²(μ'f)/∂ẋₖ∂T = −wₖ·μₖ` for the chain) ride along exactly.
"""
@views function _eval_hessian_of_lagrangian_warped(
    D::DerivativeIntegrator,
    traj::NamedTrajectory,
    μ::AbstractVector,
)
    warp = traj.warp::AbstractTimeWarp   # the _warped path is reached only under a warp
    nθ = n_params(warp)
    θ0 = warp_params(warp)
    zdim = WarpPlumbing.packed_knot_dim(traj)
    μ∂²D = spzeros(WarpPlumbing.packed_length(traj), WarpPlumbing.packed_length(traj))

    dt_first = first(traj.components[traj.timestep])
    dt_dim = traj.dims[traj.timestep]
    rows = [r for r = 1:traj.dim if !(dt_first ≤ r < dt_first + dt_dim)]
    loc(r) = r < dt_first ? r : r - dt_dim
    x_loc = [loc(r) for r in traj.components[D.x_name]]
    ẋ_loc = [loc(r) for r in traj.components[D.ẋ_name]]

    for k = 1:(traj.N-1)
        μₖ = μ[slice(k, D.x_dim)]
        cols = vcat(
            WarpPlumbing.packed_two_knot_window(traj, k),
            WarpPlumbing.warp_param_indices(traj),
        )
        μ∂²D[cols, cols] .+= ForwardDiff.hessian(
            zz -> begin
                w = zz[1:(end-nθ)]
                θ = zz[(end-nθ+1):end]
                Δtₖ = deltats(with_params(warp, θ), traj.N)[k]
                xₖ = w[x_loc]
                ẋₖ = w[ẋ_loc]
                xₖ₊₁ = w[zdim .+ x_loc]
                return μₖ'D.f(xₖ₊₁, xₖ, ẋₖ, Δtₖ)
            end,
            [traj[k].data[rows]; traj[k+1].data[rows]; θ0],
        )
    end
    return μ∂²D
end

@testitem "testing DerivativeIntegrator" begin
    include("../../test/test_utils.jl")
    traj = named_trajectory_type_1()
    D = DerivativeIntegrator(:a, :da, traj)
    test_integrator(D, traj)
end

# ============================================================================ #
# Phase 1b (DTO#149): chain rows under a time warp
# ============================================================================ #

@testitem "DerivativeIntegrator chain rows under a time warp" begin
    include("../../test/test_utils.jl")
    using NamedTrajectories
    using DirectTrajOpt: CommonInterface, WarpPlumbing, Integrators
    using TrajectoryIndexingUtils: slice, index
    using FiniteDiff
    using Test

    traj = warped_derivative_trajectory(N = 8, T = 1.7; seed = 12)
    D = DerivativeIntegrator(:u, :du, traj)
    T_col = WarpPlumbing.warp_param_indices(traj)[1]
    wₖ = 1 / (traj.N - 1)   # GlobalScale lattice weight

    # evaluation reads the DERIVED rows — unchanged code path
    δ = zeros(D.dim)
    CommonInterface.evaluate!(δ, D, traj)
    for k = 1:(traj.N-1)
        @test δ[k] ≈ traj.u[k+1] - traj.u[k] - traj.Δt[k] * traj.du[k]
    end

    # the warp column: EXACT chain entries ∂δₖ/∂T = -ẋₖ·wₖ = -duₖ/(N-1)
    ∂D = CommonInterface.eval_jacobian(D, traj)
    @test size(∂D) == (D.dim, length(traj))
    for k = 1:(traj.N-1)
        @test ∂D[k, T_col] ≈ -traj.du[k] * wₖ
    end

    # FD parity of the full Jacobian over the PACKED vector (perturbs T too)
    f̂ = Z⃗ -> begin
        t2 = copy(traj)
        unpack!(t2, Z⃗)
        δ = zeros(eltype(Z⃗), D.dim)
        CommonInterface.evaluate!(δ, D, t2)
        return δ
    end
    ∂D_fd = FiniteDiff.finite_difference_jacobian(f̂, collect(vec(traj)))
    @test all(isapprox.(∂D, ∂D_fd, atol = 1e-6, rtol = 1e-6))

    # FD parity of the Hessian of the Lagrangian
    μ = 0.3 .* collect(1.0:D.dim)
    H = CommonInterface.eval_hessian_of_lagrangian(D, traj, μ)
    ĥ = Z⃗ -> begin
        t2 = copy(traj)
        unpack!(t2, Z⃗)
        δ = zeros(eltype(Z⃗), D.dim)
        CommonInterface.evaluate!(δ, D, t2)
        return μ'δ
    end
    H_fd = FiniteDiff.finite_difference_hessian(ĥ, collect(vec(traj)))
    @test all(isapprox.(triu(H), triu(H_fd), atol = 1e-5, rtol = 1e-5))

    # the assembled structures declare the warp column (both solvers' entry paths
    # consume these — the Evaluator vcat's them directly)
    S = Integrators.get_jacobian_structure(D, traj)
    @test size(S) == (D.dim, length(traj))
    @test !iszero(S[slice(1, D.x_dim), T_col])
    HS = Integrators.get_hessian_of_lagrangian_structure(D, traj)
    @test size(HS) == (length(traj), length(traj))
    @test any(!iszero, HS[T_col, :])

    # the shared FD harness (upgrade lands with this slice) exercises both paths
    test_integrator(D, traj; atol = 1e-5)
end
