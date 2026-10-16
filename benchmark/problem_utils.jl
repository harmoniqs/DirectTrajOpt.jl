# Shared problem constructors for DirectTrajOpt benchmarks.
# Included by each @testitem via `include("$(joinpath(@__DIR__, "problem_utils.jl"))")`.

"""
    make_bilinear_problem(; N=51, seed=42)

Standard bilinear quantum-gate problem: 4D state (real Pauli representation),
2D control, with derivative and timestep integrators.
"""
function make_bilinear_problem(; N::Int = 51, seed::Int = 42)
    Random.seed!(seed)
    Δt = 0.1
    u_bound = 0.1
    ω = 0.1
    Gx = sparse(Float64[0 0 0 1; 0 0 1 0; 0 -1 0 0; -1 0 0 0])
    Gy = sparse(Float64[0 -1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 1 0])
    Gz = sparse(Float64[0 0 1 0; 0 0 0 -1; -1 0 0 0; 0 1 0 0])
    G(u) = ω * Gz + u[1] * Gx + u[2] * Gy

    traj = NamedTrajectory(
        (
            x = 2rand(4, N) .- 1,
            u = u_bound * (2rand(2, N) .- 1),
            du = randn(2, N),
            ddu = randn(2, N),
            Δt = fill(Δt, N),
        );
        controls = (:ddu, :Δt),
        timestep = :Δt,
        bounds = (u = u_bound, Δt = (0.01, 0.5)),
        initial = (x = [1.0, 0.0, 0.0, 0.0], u = zeros(2)),
        final = (u = zeros(2),),
        goal = (x = [0.0, 1.0, 0.0, 0.0],),
    )
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = QuadraticRegularizer(:u, traj, 1.0) + QuadraticRegularizer(:du, traj, 1.0)
    return DirectTrajOptProblem(traj, J, integrators)
end

"""
    make_scaled_problem(; N, state_dim, n_controls=2, seed=42)

Random bilinear problem with configurable dimensions for scaling studies.
"""
function make_scaled_problem(; N::Int, state_dim::Int, n_controls::Int = 2, seed::Int = 42)
    Random.seed!(seed)
    G_drift = sparse(randn(state_dim, state_dim))
    G_drives = [sparse(randn(state_dim, state_dim)) for _ = 1:n_controls]
    G(u) = G_drift + sum(u[i] * G_drives[i] for i = 1:n_controls)

    x_init = zeros(state_dim)
    x_init[1] = 1.0
    x_goal = zeros(state_dim)
    x_goal[min(2, state_dim)] = 1.0

    traj = NamedTrajectory(
        (
            x = randn(state_dim, N),
            u = 0.1 * randn(n_controls, N),
            du = randn(n_controls, N),
            Δt = fill(0.1, N),
        );
        controls = (:du, :Δt),
        timestep = :Δt,
        bounds = (u = 1.0, Δt = (0.01, 0.5)),
        initial = (x = x_init, u = zeros(n_controls)),
        final = (u = zeros(n_controls),),
        goal = (x = x_goal,),
    )
    integrators = [BilinearIntegrator(G, :x, :u, traj), DerivativeIntegrator(:u, :du, traj)]
    J = QuadraticRegularizer(:u, traj, 1.0)
    return DirectTrajOptProblem(traj, J, integrators)
end
