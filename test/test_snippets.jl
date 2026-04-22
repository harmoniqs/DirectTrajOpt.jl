using TestItemRunner
using TestItems

# ============================================================================
# Shared test setup via @testsnippet and @testmodule
# ============================================================================

# @testsnippet: inlined into every @testitem that lists it in setup=[...].
# Use for lightweight helper definitions (functions, constants).
# Re-evaluated per testitem — no shared mutable state.

@testsnippet DTOTestHelpers begin
    using DirectTrajOpt
    using DirectTrajOpt: Solvers, IpoptSolverExt, Callbacks
    import MathOptInterface as MOI
    import MadNLP
    import Ipopt
    using LinearAlgebra
    using SparseArrays
    using NamedTrajectories
    using Test

    # Pull in bilinear_dynamics_and_trajectory and friends from DTO's own test utils
    include(joinpath(dirname(dirname(pathof(DirectTrajOpt))), "test", "test_utils.jl"))

    # Standard problem builder used by most tests
    function make_standard_prob(; add_global=false)
        G, traj = bilinear_dynamics_and_trajectory(; add_global=add_global)
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
            times=2:(traj.N-1),
            equality=false,
        )

        prob = DirectTrajOptProblem(
            traj, J, integrators;
            constraints=AbstractConstraint[g_u_norm],
        )
        return prob, traj
    end

    # Standard problem + evaluator builder
    function make_evaluator(; eval_hessian=true)
        prob, traj = make_standard_prob()
        evaluator = Solvers.Evaluator(prob; eval_hessian=eval_hessian, verbose=false)
        return prob, traj, evaluator
    end

    # Pipe-based stdout capture for Julia 1.12+.
    # Standalone-driver workaround — drop when a better mechanism is available.
    function capture_stdout(f)
        pipe = Pipe()
        redirect_stdout(pipe) do
            f()
        end
        close(pipe.in)
        return read(pipe.out, String)
    end
end
