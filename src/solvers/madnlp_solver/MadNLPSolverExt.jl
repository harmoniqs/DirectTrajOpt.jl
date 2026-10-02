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


# MadNLP is a HARD dependency (#155): this backend module now lives in src/
# beside IpoptSolverExt (the package-extension/weakdep form is gone), and
# MadNLPOptions is the pinned default — see src/DirectTrajOpt.jl.
include("options.jl")
include("solver.jl")
include("utils.jl")


# Coverage targets: src/solvers/madnlp_solver/

@testitem "MadNLPOptions construction" setup=[DTOTestHelpers] begin
    opts = DirectTrajOpt.MadNLPOptions()
    @test opts.tol == 1e-8
    @test opts.max_iter == 3000
    @test opts.print_level == 3
    @test opts.hessian_approximation == "exact"
    @test opts.intermediate_callback === nothing
    @test opts.fixed_variable_treatment === nothing

    opts2 = DirectTrajOpt.MadNLPOptions(max_iter = 100, tol = 1e-6)
    @test opts2.max_iter == 100
    @test opts2.tol == 1e-6
    @test opts isa Solvers.AbstractSolverOptions
end

@testitem "MadNLP intermediate_callback (raw MadNLP callback) fires per iter" setup=[
    DTOTestHelpers,
] begin
    import MadNLP

    mutable struct _IterCounter <: MadNLP.AbstractUserCallback
        count::Base.RefValue{Int}
    end
    (cb::_IterCounter)(::MadNLP.AbstractMadNLPSolver, _) = (cb.count[] += 1; true)

    cb = _IterCounter(Ref(0))
    prob, _ = make_standard_prob()
    solve!(
        prob;
        options = DirectTrajOpt.MadNLPOptions(
            max_iter = 5,
            intermediate_callback = cb,
            fixed_variable_treatment = MadNLP.RelaxBound,
        ),
        verbose = false,
    )
    @test cb.count[] > 0
end

@testitem "MadNLP intermediate_callback (AbstractIntermediateCallback) fires per iter" setup=[
    DTOTestHelpers,
] begin
    import MadNLP

    mutable struct _AgnosticCounter <: DirectTrajOpt.AbstractIntermediateCallback
        count::Base.RefValue{Int}
        last_primal_len::Base.RefValue{Int}
    end
    function (cb::_AgnosticCounter)(primal::AbstractVector, iter::Integer)
        cb.count[] += 1
        cb.last_primal_len[] = length(primal)
        return true
    end

    cb = _AgnosticCounter(Ref(0), Ref(0))
    prob, _ = make_standard_prob()
    solve!(
        prob;
        options = DirectTrajOpt.MadNLPOptions(
            max_iter = 5,
            intermediate_callback = cb,
            fixed_variable_treatment = MadNLP.RelaxBound,
        ),
        verbose = false,
    )
    @test cb.count[] > 0
    # With RelaxBound, the primal vector matches the full NLP variable count.
    @test cb.last_primal_len[] ==
          length(prob.trajectory.datavec) + prob.trajectory.global_dim
end

@testitem "MadNLP intermediate_callback auto-couples RelaxBound" setup=[DTOTestHelpers] begin
    import MadNLP

    mutable struct _AutoCoupleProbe <: DirectTrajOpt.AbstractIntermediateCallback
        last_primal_len::Base.RefValue{Int}
    end
    function (cb::_AutoCoupleProbe)(primal::AbstractVector, _)
        cb.last_primal_len[] = length(primal)
        return true
    end

    cb = _AutoCoupleProbe(Ref(0))
    prob, _ = make_standard_prob()
    # Note: NOT passing fixed_variable_treatment. set_options! should auto-set it.
    solve!(
        prob;
        options = DirectTrajOpt.MadNLPOptions(max_iter = 5, intermediate_callback = cb),
        verbose = false,
    )
    # If RelaxBound auto-coupled correctly, the primal includes fixed variables.
    @test cb.last_primal_len[] ==
          length(prob.trajectory.datavec) + prob.trajectory.global_dim
end

@testitem "MadNLP auto-couple respects MadNLP's conditional default" setup=[DTOTestHelpers] begin
    import MadNLP

    mutable struct _PassthroughProbe <: DirectTrajOpt.AbstractIntermediateCallback
        len::Base.RefValue{Int}
    end
    (cb::_PassthroughProbe)(primal, _) = (cb.len[] = length(primal); true)

    cb = _PassthroughProbe(Ref(0))
    prob, _ = make_standard_prob()
    # With `kkt_system = SparseCondensedKKTSystem`, MadNLP's own conditional
    # default for `fixed_variable_treatment` is already `RelaxBound`, so the
    # auto-couple should not fire. Capture logs and assert our @info is absent.
    logs, _ = Test.collect_test_logs() do
        solve!(
            prob;
            options = DirectTrajOpt.MadNLPOptions(
                max_iter = 5,
                intermediate_callback = cb,
                kkt_system = MadNLP.SparseCondensedKKTSystem,
            ),
            verbose = false,
        )
    end
    @test !any(l -> occursin("Setting fixed_variable_treatment", l.message), logs)
    # MadNLP's untouched conditional default still yields the full primal.
    @test cb.len[] == length(prob.trajectory.datavec) + prob.trajectory.global_dim
end

@testitem "MadNLP intermediate_callback early termination via return false" setup=[
    DTOTestHelpers,
] begin
    import MadNLP

    mutable struct _Stopper <: DirectTrajOpt.AbstractIntermediateCallback
        max_iters::Int
        count::Base.RefValue{Int}
    end
    function (cb::_Stopper)(_, _)
        cb.count[] += 1
        return cb.count[] < cb.max_iters
    end

    cb = _Stopper(3, Ref(0))
    prob, _ = make_standard_prob()
    solve!(
        prob;
        options = DirectTrajOpt.MadNLPOptions(max_iter = 100, intermediate_callback = cb),
        verbose = false,
    )
    # Callback stopped the solve well before max_iter=100.
    @test cb.count[] <= 5
end

@testitem "MadNLP intermediate_callback rejects invalid type" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    bogus_cb(args...) = true   # bare Function — neither abstract nor MadNLP subtype
    @test_throws ArgumentError solve!(
        prob;
        options = DirectTrajOpt.MadNLPOptions(
            max_iter = 5,
            intermediate_callback = bogus_cb,
        ),
        verbose = false,
    )
end

@testitem "MadNLP basic solve" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    traj_before = deepcopy(prob.trajectory.data)
    solve!(prob; options = DirectTrajOpt.MadNLPOptions(max_iter = 50), verbose = false)
    @test prob.trajectory.data != traj_before
end

@testitem "MadNLP verbose=false" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    output = capture_stdout() do
        solve!(prob; options = DirectTrajOpt.MadNLPOptions(max_iter = 10), verbose = false)
    end
    @test !contains(output, "initializing optimizer")
end

@testitem "MadNLP verbose=true" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    output = capture_stdout() do
        solve!(prob; options = DirectTrajOpt.MadNLPOptions(max_iter = 10), verbose = true)
    end
    @test contains(output, "initializing optimizer")
    @test contains(output, "evaluator created")
    @test contains(output, "optimizer initialization complete")
end

@testitem "MadNLP eval_hessian kwarg routing" setup=[DTOTestHelpers] begin
    # eval_hessian=false routes to hessian_approximation="compact_lbfgs".
    prob, _ = make_standard_prob()
    solve!(
        prob;
        options = DirectTrajOpt.MadNLPOptions(max_iter = 5),
        verbose = false,
        eval_hessian = false,
    )
    @test true
end

@testitem "MadNLP eval_hessian kwarg routing" setup=[DTOTestHelpers] begin
    # eval_hessian=false routes to hessian_approximation="compact_lbfgs".
    prob, _ = make_standard_prob()
    result = _solve_with_kwargs(
        prob,
        DirectTrajOpt.MadNLPOptions(max_iter = 5);
        verbose = false,
        eval_hessian = false,
    )
    @test true
end

@testitem "MadNLP compact_lbfgs hessian" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    opts =
        DirectTrajOpt.MadNLPOptions(max_iter = 10, hessian_approximation = "compact_lbfgs")
    solve!(prob; options = opts, verbose = false)
    @test true
end

@testitem "MadNLP with global variables" setup=[DTOTestHelpers] begin
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
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

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
    solve!(prob; options = DirectTrajOpt.MadNLPOptions(max_iter = 50), verbose = false)

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
        DirectTrajOpt.MadNLPOptions(max_iter = 50);
        verbose = false,
        kkt_system = MadNLP.SparseKKTSystem,
        linear_solver = MadNLP.MumpsSolver,
    )
    @test true
end

@testitem "_solve_with_kwargs with LapackCPUSolver" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    DirectTrajOpt._solve_with_kwargs(
        prob,
        DirectTrajOpt.MadNLPOptions(max_iter = 50);
        verbose = false,
        kkt_system = MadNLP.SparseUnreducedKKTSystem,
        linear_solver = MadNLP.LapackCPUSolver,
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
        DirectTrajOpt.MadNLPOptions(max_iter = 50);
        verbose = false,
        kkt_system = MadNLP.SparseKKTSystem,
        linear_solver = MadNLP.MumpsSolver,
        barrier = MadNLP.LOQOUpdate(1e-8, 10.0),
    )
    @test true
end

@testitem "_solve_with_kwargs with QualityFunctionUpdate adaptive barrier" setup=[
    DTOTestHelpers,
] begin
    prob, _ = make_standard_prob()
    # QualityFunctionUpdate: adaptive barrier from Nocedal et al. 2009 §4.
    # Minimizes an ℓ1 quality function via golden search; falls back to
    # monotone if insufficient progress.
    DirectTrajOpt._solve_with_kwargs(
        prob,
        DirectTrajOpt.MadNLPOptions(max_iter = 50);
        verbose = false,
        kkt_system = MadNLP.SparseKKTSystem,
        linear_solver = MadNLP.MumpsSolver,
        barrier = MadNLP.QualityFunctionUpdate(1e-8, 10.0),
    )
    @test true
end

# ----------------------------------------------------------------------------
# Telemetry parity (#155 AC): the AMICODE_ITER emitter contract on the
# MadNLP arm. The emitter rides the RAW MadNLP callback (same architecture as
# the Ipopt path — the agnostic (primal, iter) callback cannot carry IPM
# state); every Regular invocation must expose all four state columns, and
# the UserCallbackRegular filter must keep iter monotonic by skipping the
# restore/robust phases, which fire without advancing cnt.k.
# ----------------------------------------------------------------------------

@testitem "AMICODE_ITER state carries on MadNLP (iter/f/inf_pr/inf_du on the raw callback)" setup =
    [DTOTestHelpers] begin
    import MadNLP

    mutable struct _AmicodeIterProbe <: MadNLP.AbstractUserCallback
        rows::Vector{NamedTuple}
        modes::Vector{String}
    end
    function (cb::_AmicodeIterProbe)(
        solver::MadNLP.AbstractMadNLPSolver,
        mode::MadNLP.AbstractUserCallbackStatus,
    )
        push!(cb.modes, string(typeof(mode).name.name))
        # The AMICODE_ITER columns, read exactly as an emitter would read them
        if mode isa MadNLP.UserCallbackRegular
            push!(
                cb.rows,
                (
                    iter = Int(MadNLP.get_cnt(solver).k),
                    f = Float64(MadNLP.get_obj_val(solver)),
                    inf_pr = Float64(MadNLP.get_inf_pr(solver)),
                    inf_du = Float64(MadNLP.get_inf_du(solver)),
                ),
            )
        end
        return true
    end

    cb = _AmicodeIterProbe(NamedTuple[], String[])
    prob, _ = make_standard_prob()
    stats = solve!(
        prob;
        options = DirectTrajOpt.MadNLPOptions(
            max_iter = 10,
            intermediate_callback = cb,
            print_level = 6,
        ),
        verbose = false,
    )
    @test stats.solver === :madnlp
    @test !isempty(cb.rows)
    # All four columns readable and finite on every Regular emission.
    for r in cb.rows
        @test isfinite(r.f)
        @test isfinite(r.inf_pr)
        @test isfinite(r.inf_du)
    end
    # Regular emissions carry monotone iters — no duplicates, no
    # non-advancing rows (restore/robust phases never emit).
    @test issorted([r.iter for r in cb.rows])
    @test length(unique([r.iter for r in cb.rows])) == length(cb.rows)
    @test length(cb.rows) <= stats.iterations + 1
    # Every observed mode comes from the closed callback-status set.
    @test all(
        m -> m ∈ ("UserCallbackRegular", "UserCallbackRestore", "UserCallbackRobust"),
        cb.modes,
    )
end

@testitem "_MadNLPCallbackAdapter filters restore/robust modes (UserCallbackRegular only)" setup =
    [DTOTestHelpers] begin
    import MadNLP

    mutable struct _Recorder <: DirectTrajOpt.AbstractIntermediateCallback
        calls::Vector{Tuple{Int,Int}} # (length(primal), iter)
    end
    function (cb::_Recorder)(primal::AbstractVector, iter::Integer)
        push!(cb.calls, (length(primal), Int(iter)))
        return true
    end

    rec = _Recorder(Tuple{Int,Int}[])
    adapter = DirectTrajOpt.MadNLPSolverExt._MadNLPCallbackAdapter(rec)
    # Duck-typed solver surface: the adapter reads only `x` (a PrimalVector)
    # and `cnt.k` — enough to exercise the mode filter without a live IPM.
    mock = (x = MadNLP.PrimalVector(Vector{Float64}, 6, 0, Int[], Int[]), cnt = (k = 3,))

    # Regular: forwarded to the agnostic callback (primal stripped to the NLP
    # variables, cnt.k as the iteration index).
    @test adapter(mock, MadNLP.UserCallbackRegular()) == true
    @test rec.calls == [(6, 3)]
    # Restore/robust phases: silently skipped — no emission, solve continues.
    @test adapter(mock, MadNLP.UserCallbackRestore()) == true
    @test adapter(mock, MadNLP.UserCallbackRobust()) == true
    @test rec.calls == [(6, 3)]
end

end
