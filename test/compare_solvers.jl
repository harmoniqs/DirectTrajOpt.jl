import Random
import Ipopt
import MadNLP

using LinearAlgebra
using SparseArrays
using NamedTrajectories
using DirectTrajOpt

const MadNLPSolverExt =
    [mod for mod in reverse(Base.loaded_modules_order) if Symbol(mod) == :MadNLPSolverExt][1]

function get_seeded_trajectory(seed; N = 10, Δt = 0.1, u_bound = 0.1, ω = 0.1)
    Random.seed!(seed)

    Gx = sparse(Float64[
        0 0 0 1;
        0 0 1 0;
        0 -1 0 0;
        -1 0 0 0
    ])

    Gy = sparse(Float64[
        0 -1 0 0;
        1 0 0 0;
        0 0 0 -1;
        0 0 1 0
    ])

    Gz = sparse(Float64[
        0 0 1 0;
        0 0 0 -1;
        -1 0 0 0;
        0 1 0 0
    ])

    G_drift = Gz
    G_drives = [Gx, Gy]

    G(u) = ω * G_drift + sum(u .* G_drives)

    u_initial = u_bound * (2rand(2, N) .- 1)
    x_initial = 2rand(4, N) .- 1

    x_init = [1.0, 0.0, 0.0, 0.0]
    x_goal = [0.0, 1.0, 0.0, 0.0]

    traj = NamedTrajectory(
        (
            x = x_initial,
            u = u_initial,
            du = randn(2, N),
            ddu = randn(2, N),
            Δt = fill(Δt, N),
        );
        controls = (:ddu, :Δt),
        timestep = :Δt,
        bounds = (u = (-u_bound, u_bound), Δt = (1.0, 1.0)), # timestep variability is a major source of error as in the "multiple comparisons problem" so we make them constant here
        initial = (x = x_init, u = zeros(2)),
        final = (u = zeros(2),),
        goal = (x = x_goal,),
    )

    return G, traj
end

function get_ipopt_traj(seed)
    G, traj = get_seeded_trajectory(seed)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    g_u_norm = NonlinearKnotPointConstraint(
        u -> [norm(u) - 1.0],
        :u,
        traj;
        times = 2:(traj.N-1),
        equality = false,
    )

    prob = DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints = AbstractConstraint[g_u_norm],
    )

    solve!(prob; options = IpoptSolverExt.IpoptOptions(; max_iter = 100))

    return prob.trajectory
end

function get_madnlp_traj(seed)
    G, traj = get_seeded_trajectory(seed)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    g_u_norm = NonlinearKnotPointConstraint(
        u -> [norm(u) - 1.0],
        :u,
        traj;
        times = 2:(traj.N-1),
        equality = false,
    )

    prob = DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints = AbstractConstraint[g_u_norm],
    )

    solve!(prob; options = MadNLPOptions(; max_iter = 100))

    return prob.trajectory
end

function get_solver_comparison(seed)
    ti = @elapsed (di = get_ipopt_traj(seed).data[:, :])
    tm = @elapsed (dm = get_madnlp_traj(seed).data[:, :])
    dd = ((dm .- di) .^ 2)
    err = sqrt(sum(dd) / length(dd))
    return err, (ti, tm)
end

wins = Dict(:ipopt => 0, :madnlp => 0)
for seed = 0:99
    err, (ti, tm) = get_solver_comparison(seed)
    (err < 1e-3) || exit(1)
    wins[(ti < tm) ? :ipopt : :madnlp] += 1
end

# @info "Wins: $(wins)"
