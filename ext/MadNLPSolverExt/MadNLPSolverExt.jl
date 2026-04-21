module MadNLPSolverExt

import MathOptInterface as MOI
import MadNLP # DO NOT using!

using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils

using TestItemRunner


using DirectTrajOpt.Constraints
using DirectTrajOpt.Integrators
using DirectTrajOpt.Objectives
using DirectTrajOpt.Solvers


# include("options.jl") # moved to solvers/madnlp_solver/options.jl
include("solver.jl")
include("utils.jl")


# Coverage targets: ext/MadNLPSolverExt/ + src/solvers/madnlp_solver/

@testitem "MadNLPOptions construction" setup=[DTOTestHelpers] begin
    opts = DirectTrajOpt.MadNLPOptions()
    @test opts.tol == 1e-8
    @test opts.max_iter == 3000
    @test opts.print_level == 3
    @test opts.hessian_approximation == "exact"

    opts2 = DirectTrajOpt.MadNLPOptions(max_iter=100, tol=1e-6)
    @test opts2.max_iter == 100
    @test opts2.tol == 1e-6
    @test opts isa Solvers.AbstractSolverOptions
end

@testitem "MadNLP basic solve" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    traj_before = deepcopy(prob.trajectory.data)
    solve!(prob; options=DirectTrajOpt.MadNLPOptions(max_iter=50), verbose=false)
    @test prob.trajectory.data != traj_before
end

@testitem "MadNLP verbose=false" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    output = capture_stdout() do
        solve!(prob; options=DirectTrajOpt.MadNLPOptions(max_iter=10), verbose=false)
    end
    @test !contains(output, "initializing optimizer")
end

@testitem "MadNLP verbose=true" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    output = capture_stdout() do
        solve!(prob; options=DirectTrajOpt.MadNLPOptions(max_iter=10), verbose=true)
    end
    @test contains(output, "initializing optimizer")
    @test contains(output, "evaluator created")
    @test contains(output, "optimizer initialization complete")
end

# @testitem "MadNLP unknown kwargs passthrough" setup=[DTOTestHelpers] begin
#     # On this branch the @warn for unknown kwargs is commented out in
#     # ext/MadNLPSolverExt/solver.jl. Unknown kwargs are silently collected.
#     # When warnings are re-enabled, add @test_logs assertion here.
#     prob, _ = make_standard_prob()
#     solve!(prob; options=DirectTrajOpt.MadNLPOptions(max_iter=5), verbose=false, totally_fake_option=42)
#     @test true
# end

@testitem "MadNLP eval_hessian kwarg routing" setup=[DTOTestHelpers] begin
    # eval_hessian=false routes to hessian_approximation="compact_lbfgs".
    # The @warn is commented out on this branch — just verify no error.
    prob, _ = make_standard_prob()
    solve!(prob; options=DirectTrajOpt.MadNLPOptions(max_iter=5), verbose=false, eval_hessian=false)
    @test true
end

@testitem "MadNLP eval_hessian kwarg routing" setup=[DTOTestHelpers] begin
    # eval_hessian=false routes to hessian_approximation="compact_lbfgs".
    # The @warn is commented out on this branch — just verify no error.
    prob, _ = make_standard_prob()
    result = _solve_with_kwargs(prob, DirectTrajOpt.MadNLPOptions(max_iter=5); verbose=false, eval_hessian=false)
    @test true
end

@testitem "MadNLP compact_lbfgs hessian" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    opts = DirectTrajOpt.MadNLPOptions(max_iter=10, hessian_approximation="compact_lbfgs")
    solve!(prob; options=opts, verbose=false)
    @test true
end

@testitem "MadNLP with global variables" setup=[DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory(add_global=true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q=1.0)

    g_ug = NonlinearGlobalKnotPointConstraint(
        ug -> begin
            u = ug[1:traj.dims[:u]]
            g = ug[(traj.dims[:u]+1):end]
            return [norm(u) * (1.0 + norm(g)) - 2.0]
        end,
        [:u], [:g], traj;
        times=2:(traj.N-1), equality=false,
    )
    prob = DirectTrajOptProblem(traj, J, integrators; constraints=AbstractConstraint[g_ug])
    solve!(prob; options=DirectTrajOpt.MadNLPOptions(max_iter=50), verbose=false)

    for k = 2:(traj.N-1)
        u = traj[k][:u]
        g = traj.global_data[traj.global_components[:g]]
        @test norm(u) * (1.0 + norm(g)) <= 2.0 + 1e-5
    end
end

@testitem "_solve_with_kwargs with MumpsSolver (default)" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    DirectTrajOpt._solve_with_kwargs(
        prob,
        DirectTrajOpt.MadNLPOptions(max_iter=50);
        verbose=false,
        kkt_system=MadNLP.SparseKKTSystem,
        linear_solver=MadNLP.MumpsSolver,
    )
    @test true
end

@testitem "_solve_with_kwargs with LapackCPUSolver" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    DirectTrajOpt._solve_with_kwargs(
        prob,
        DirectTrajOpt.MadNLPOptions(max_iter=50);
        verbose=false,
        kkt_system=MadNLP.SparseUnreducedKKTSystem,
        linear_solver=MadNLP.LapackCPUSolver,
    )
    @test true
end

@testitem "_solve_with_kwargs with LOQOUpdate adaptive barrier" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    # LOQOUpdate: adaptive barrier from Nocedal et al. 2009 §3.
    # Uses average/min complementarity ratio to set the barrier parameter.
    # Falls back to monotone if insufficient progress.
    DirectTrajOpt._solve_with_kwargs(
        prob,
        DirectTrajOpt.MadNLPOptions(max_iter=50);
        verbose=false,
        kkt_system=MadNLP.SparseKKTSystem,
        linear_solver=MadNLP.MumpsSolver,
        barrier=MadNLP.LOQOUpdate(1e-8, 10.0),
    )
    @test true
end

@testitem "_solve_with_kwargs with QualityFunctionUpdate adaptive barrier" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    # QualityFunctionUpdate: adaptive barrier from Nocedal et al. 2009 §4.
    # Minimizes an ℓ1 quality function via golden search; falls back to
    # monotone if insufficient progress.
    DirectTrajOpt._solve_with_kwargs(
        prob,
        DirectTrajOpt.MadNLPOptions(max_iter=50);
        verbose=false,
        kkt_system=MadNLP.SparseKKTSystem,
        linear_solver=MadNLP.MumpsSolver,
        barrier=MadNLP.QualityFunctionUpdate(1e-8, 10.0),
    )
    @test true
end

end
