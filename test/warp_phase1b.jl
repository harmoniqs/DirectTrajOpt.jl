# ============================================================================
# Phase 1b (DTO#149) — warp-aware evaluator, end to end (DTO side).
#
# The scope amendment re-targets this issue: BilinearIntegrator is on the
# demotion arc (no warp plumbing), the warp-consuming dynamics integrator is
# Piccolo's HermitianExponentialIntegrator (Piccolo.jl#321, parallel slice).
# Until that cell merges, the end-to-end AC relaxes to DTO-side evaluator
# parity + FD tests, run here: a warp+uniform-mesh free-duration problem —
# ONE duration variable T (the GlobalScale warp parameter) replacing the
# per-knot free-Δt block — assembled through the DTO evaluator and solved
# through the packed Ipopt path.
#
# NOT run here (stated for the record): the drift-dominated 1q X-gate
# benchmark regression — it needs the surviving dynamics integrator
# (Piccolo.jl#321, open) and runs when that slice lands.
# ============================================================================ #

@testitem "Phase 1b end-to-end: warp+uniform-mesh free-duration problem through the Evaluator" begin
    include(joinpath(@__DIR__, "test_utils.jl"))
    using NamedTrajectories
    using DirectTrajOpt
    using DirectTrajOpt: Solvers, WarpPlumbing, CommonInterface
    using DirectTrajOpt.WarpPlumbing: packed_slice, warp_param_indices, packed_row_index
    using TrajectoryIndexingUtils: slice, index
    using FiniteDiff
    using SparseArrays
    using Test
    import MathOptInterface as MOI

    N = 10
    T0 = 1.0
    traj = NamedTrajectory(
        (
            x = randn(2, N),
            u = randn(1, N),
            du = randn(1, N),
            Δt = fill(T0 / (N - 1), 1, N),   # ignored — derived from the warp
        );
        controls = (:u, :du),
        timestep = :Δt,
        warp = GlobalScale(T0),
        initial = (x = [0.0, 0.0],),
        final = (x = [1.0, 1.0],),
        goal = (x = [1.0, 1.0],),
        bounds = (u = (-1.0, 1.0), du = (-5.0, 5.0)),
    )

    integrators = AbstractIntegrator[DerivativeIntegrator(:u, :du, traj)]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += LinearRegularizer(:du, traj, 0.1)
    J += MinimumTimeObjective(traj, D = 0.5)

    g_u_norm = NonlinearKnotPointConstraint(
        u -> [norm(u) - 1.0],
        :u,
        traj;
        times = 2:(N-1),
        equality = false,
    )
    # the free duration needs a floor — the ONE duration variable's box
    T_bounds = WarpParamBoundsConstraint([0.2], [5.0])

    prob = DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints = AbstractConstraint[g_u_norm, T_bounds],
    )

    evaluator = Solvers.Evaluator(prob; eval_hessian = true, verbose = false)

    n_packed = length(vec(traj))
    T_col = warp_param_indices(traj)[1]
    z0 = collect(vec(traj))

    # packed accounting: structures live in packed coordinates and DECLARE the
    # warp column (the dynamics rows chain through Δtₖ(T))
    @test maximum(last, evaluator.jacobian_structure) ≤ n_packed
    @test any(last.(evaluator.jacobian_structure) .== T_col)
    @test maximum(first, evaluator.hessian_structure) ≤ n_packed &&
          maximum(last, evaluator.hessian_structure) ≤ n_packed
    @test any(rs -> T_col in rs, last.(evaluator.hessian_structure))

    # objective + gradient FD parity over the PACKED vector
    @test MOI.eval_objective(evaluator, z0) ≈ objective_value(J, traj)
    ∇ = zeros(n_packed)
    MOI.eval_objective_gradient(evaluator, ∇, z0)
    ∇_fd = FiniteDiff.finite_difference_gradient(Z⃗ -> MOI.eval_objective(evaluator, Z⃗), z0)
    @test all(isapprox.(∇, ∇_fd, atol = 1e-5, rtol = 1e-5))

    # constraint values + Jacobian FD parity
    ĝ = Z⃗ -> begin
        g = zeros(eltype(Z⃗), evaluator.n_constraints)
        MOI.eval_constraint(evaluator, g, Z⃗)
        return g
    end
    g = zeros(evaluator.n_constraints)
    MOI.eval_constraint(evaluator, g, z0)
    @test g ≈ ĝ(z0)

    ∂g_vals = zeros(length(evaluator.jacobian_structure))
    MOI.eval_constraint_jacobian(evaluator, ∂g_vals, z0)
    ∂g = zeros(evaluator.n_constraints, n_packed)
    for (v, (i, j)) in zip(∂g_vals, evaluator.jacobian_structure)
        ∂g[i, j] = v
    end
    ∂g_fd = FiniteDiff.finite_difference_jacobian(ĝ, z0)
    @test all(isapprox.(∂g_fd, ∂g, atol = 1e-5, rtol = 1e-5))

    # Hessian of the Lagrangian FD parity
    μ = 0.1 .* ones(evaluator.n_constraints)
    σ = 2.0
    ∂²ℒ_vals = zeros(length(evaluator.hessian_structure))
    MOI.eval_hessian_lagrangian(evaluator, ∂²ℒ_vals, z0, σ, μ)
    ∂²ℒ = spzeros(n_packed, n_packed)
    for (v, (i, j)) in zip(∂²ℒ_vals, evaluator.hessian_structure)
        ∂²ℒ[i, j] = v
    end
    ∂²ℒ_fd = FiniteDiff.finite_difference_hessian(z0) do Z⃗
        return σ * MOI.eval_objective(evaluator, Z⃗) + μ'ĝ(Z⃗)
    end
    @test all(isapprox.(triu(∂²ℒ), triu(sparse(∂²ℒ_fd)), atol = 1e-3, rtol = 1e-3))

    # Ipopt end-to-end through the packed path: T is the only free time quantity,
    # the solve writes back through unpack!, and the derived rows stay synced
    solve!(prob; max_iter = 200, verbose = false, print_level = 0)
    T_solved = prob.trajectory.warp.T
    @test 0.2 ≤ T_solved ≤ T0 + 1e-6          # min-time drives T down to the floor
    @test prob.trajectory.Δt ≈ fill(T_solved / (N - 1), 1, N)
    @test all(k -> prob.trajectory[k].timestep ≈ T_solved / (N - 1), 1:N)
end
