using DirectTrajOpt

function get_prob()
    include("./DirectTrajOpt.jl/test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

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

function get_prob_and_tests()
    include("./DirectTrajOpt.jl/test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Add global objective - minimize global parameter
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

    # Knot point constraint with global dependency
    # Couples control magnitude with global parameter
    g_ug = NonlinearGlobalKnotPointConstraint(
        ug -> begin
            u = ug[1:traj.dims[:u]]
            g = ug[(traj.dims[:u]+1):end]
            return [norm(u) * (1.0 + norm(g)) - 2.0]
        end,
        [:u],
        [:g],
        traj;
        times = 2:(traj.N-1),
        equality = false,
    )

    prob =
        DirectTrajOptProblem(traj, J, integrators; constraints = AbstractConstraint[g_ug])

    # solve!(prob; max_iter = 100)

    # Verify constraint is satisfied at each timestep
    testprobfn() =
        for k = 2:(traj.N-1)
            u = traj[k][:u]
            g = traj.global_data[traj.global_components[:g]]
            @assert norm(u) * (1.0 + norm(g)) <= 2.0 + 1e-6
        end

    return prob, testprobfn
end
