# Cross-backend comparison harness (#155): the DEFAULT leg (no options —
# rides the pinned default, MadNLP since the flip) and the explicit Ipopt leg
# must land on the same trajectory from the same seed. The explicit MadNLP
# leg stays for the day the default moves again — it pins the named backend
# independently of whatever `_DefaultSolverOptions` points at.

@testsnippet DTOCompareSolvers begin
    import Random

    using DirectTrajOpt
    using DirectTrajOpt: IpoptSolverExt, MadNLPOptions
    using LinearAlgebra
    using SparseArrays
    using NamedTrajectories

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
            # timestep variability is a major source of error as in the
            # "multiple comparisons problem" so we make them constant here
            bounds = (u = (-u_bound, u_bound), Δt = (1.0, 1.0)),
            initial = (x = x_init, u = zeros(2)),
            final = (u = zeros(2),),
            goal = (x = x_goal,),
        )

        return G, traj
    end

    function get_seeded_prob(seed)
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

        return prob
    end

    # ── The legs ──────────────────────────────────────────────────────────

    # Explicit Ipopt leg — the demoted-but-fully-selectable backend.
    function get_ipopt_traj(seed)
        prob = get_seeded_prob(seed)
        solve!(prob; options = IpoptSolverExt.IpoptOptions(; max_iter = 100))
        return prob.trajectory
    end

    # Explicit MadNLP leg — pins the named backend, default-independent.
    function get_madnlp_traj(seed)
        prob = get_seeded_prob(seed)
        solve!(prob; options = MadNLPOptions(; max_iter = 100))
        return prob.trajectory
    end

    # Default leg — no options kwarg; rides whatever `_DefaultSolverOptions`
    # is pinned to.
    function get_default_traj(seed)
        prob = get_seeded_prob(seed)
        stats = solve!(prob; max_iter = 100)
        return prob.trajectory, stats
    end

    # ── Comparison driver ─────────────────────────────────────────────────

    # Run the default leg and the Ipopt leg on the same seeded problem;
    # return the RMS trajectory distance between their solutions, the pair of
    # leg walls, and the backend symbol the default leg actually dispatched.
    function get_solver_comparison(seed)
        td = @elapsed (dd, stats) = get_default_traj(seed)
        ti = @elapsed (di = get_ipopt_traj(seed).data[:, :])
        dd = dd.data[:, :]
        dist = ((dd .- di) .^ 2)
        err = sqrt(sum(dist) / length(dist))
        return err, (td, ti), stats.solver
    end
end

@testitem "compare_solvers: the default leg (MadNLP) and the Ipopt leg agree" setup =
    [DTOTestHelpers, DTOCompareSolvers] begin
    # Fixed seeds: the comparison is deterministic and reproducible — a
    # regression on any seed is a real solver divergence, not a draw.
    # Threshold 1e-3 on the RMS distance (this file's historical driver
    # threshold): both arms CONVERGE here (tol 1e-8 each), and two
    # independently-converged iterates of this non-strictly-determined
    # optimum sit within a ~5e-4 RMS tolerance cone of each other — the
    # metric measures agreement of the KKT neighborhood, not bit equality.
    for seed = 1:3
        err, _, default_solver = get_solver_comparison(seed)
        @test default_solver === :madnlp
        @test err < 1e-3
    end
end

@testitem "compare_solvers: the explicit MadNLP leg matches the Ipopt leg" setup =
    [DTOTestHelpers, DTOCompareSolvers] begin
    # Same RMS tolerance-cone rationale as the default-leg comparison above.
    for seed = 1:3
        dm = get_madnlp_traj(seed).data[:, :]
        di = get_ipopt_traj(seed).data[:, :]
        dist = ((dm .- di) .^ 2)
        @test sqrt(sum(dist) / length(dist)) < 1e-3
    end
end
