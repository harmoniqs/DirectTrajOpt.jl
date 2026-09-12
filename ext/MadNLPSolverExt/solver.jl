using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils

# using MathOptInterface
# const MOI = MathOptInterface
import MathOptInterface as MOI
import MadNLP # DO NOT using!
using TestItemRunner
# using Libdl  # Added for Pardiso library loading


function DirectTrajOpt._solve(
    prob::DirectTrajOptProblem,
    options::MadNLPOptions;
    verbose::Bool = true,
    callback = nothing,
    kwargs...,
)
    # Apply kwargs to matching MadNLPOptions fields. Unmatched kwargs warn
    # loudly (the ipopt_solver/solver.jl convention — zero silent drops);
    # the MadNLP-native raw pass-through lives on `_solve_with_kwargs`.
    madnlp_fields = fieldnames(MadNLPOptions)
    for (k, v) in kwargs
        if k in madnlp_fields
            setfield!(options, k, v)
        elseif k !== :eval_hessian
            @warn "Unknown solver option: $k. Valid options: $(madnlp_fields)"
        end
    end

    # Sync derived fields that depend on other fields. (Keyed reads only:
    # on Julia 1.12 `kwargs` is a `Base.Pairs` — property access throws.)
    eval_hessian = get(kwargs, :eval_hessian, nothing)
    if eval_hessian !== nothing
        setfield!(options, :hessian_approximation, eval_hessian ? "exact" : "compact_lbfgs")
    end

    # Instantiate MadNLP.Optimizer <: MOI.AbstractOptimizer
    #   1. Set MOI.NLPBlock()
    #   2. Set MOI.ObjectiveSense()
    #   3. Set MOI.VariablePrimal()
    #   4. TODO: Set MOI.NLPBlockDualStart() (optional)
    #   5. TODO: Set callbacks (optional)
    #   6. Add linear constraints
    #   7. Set optimizer options (involves conversions of the form convert(k::Symbol, v_in::Union{Real, String}, v_out::Any), where some of the v_out types are internal to MadNLP)

    optimizer, variables =
        get_optimizer_and_variables(prob, options, callback, verbose = verbose)

    t_solve = time()
    try
        MOI.optimize!(optimizer)

        # TODO: Verify this is working as expected
        update_trajectory!(prob, optimizer, variables)

        return Solvers._solve_stats(optimizer, variables, :madnlp, t_solve)
    catch err
        # Failure path: the benchmark/CI walls read ONE SolveStats contract —
        # a thrown solve returns populated stats with a failure-class status
        # instead of propagating. The trajectory is left as-is: no primal is
        # guaranteed to exist. (Status-based failures — ITERATION_LIMIT and
        # friends — return normally through `_solve_stats` above.)
        @warn "MadNLP solve! failed; returning failure SolveStats" exception =
            (err, catch_backtrace())
        return _madnlp_failure_stats(optimizer, err, t_solve)
    end
end

"""
    _madnlp_failure_stats(optimizer, err, t_solve) -> Solvers.SolveStats

Best-effort `SolveStats` from a FAILED MadNLP solve. A thrown solve leaves
MadNLP's MOI `result` unset (`nothing`), so every result-backed MOI getter can
throw — each read is guarded and falls back to a safe placeholder
(`MOI.OTHER_ERROR`, the exception's message, `NaN`, `-1`).
"""
function _madnlp_failure_stats(optimizer, err, t_solve::Float64)
    status = try
        MOI.get(optimizer, MOI.TerminationStatus())
    catch
        MOI.OTHER_ERROR
    end
    raw = try
        MOI.get(optimizer, MOI.RawStatusString())
    catch
        err isa Exception ? sprint(showerror, err) : string(status)
    end
    obj = try
        Float64(MOI.get(optimizer, MOI.ObjectiveValue()))
    catch
        NaN
    end
    iters = try
        Int(MOI.get(optimizer, MOI.BarrierIterations()))
    catch
        -1
    end
    return Solvers.SolveStats(;
        status = status,
        raw_status = raw,
        objective_value = obj,
        iterations = iters,
        solve_time_s = time() - t_solve,
        solver = :madnlp,
    )
end


# ----------------------------------------------------------------------------
# Optimizer Initialization/Synchronization
# ----------------------------------------------------------------------------


function get_optimizer_and_variables(
    prob::DirectTrajOptProblem,
    options::MadNLPOptions,
    callback::Union{Nothing,Function};
    verbose::Bool = true,
)
    t_init_start = time()
    if verbose
        println("    initializing optimizer...")
    end

    # get evaluator
    t_eval = time()
    evaluator = Solvers.Evaluator(prob; eval_hessian = true, verbose = verbose)
    if verbose
        println("    evaluator created ($(round(time() - t_eval, digits=3))s)")
    end

    # get the MOI specific nonlinear constraints
    t_nlcons = time()
    nl_cons = Solvers.get_nonlinear_constraints(prob)
    if verbose
        println(
            "    NL constraint bounds extracted ($(round(time() - t_nlcons, digits=3))s)",
        )
    end

    # build NLP block data
    t_block = time()
    block_data = MOI.NLPBlockData(nl_cons, evaluator, true)
    if verbose
        println("    NLP block data built ($(round(time() - t_block, digits=3))s)")
    end

    # initialize optimizer 
    t_opt = time()
    optimizer = MadNLP.Optimizer()

    # set NLP block data
    MOI.set(optimizer, MOI.NLPBlock(), block_data)

    # set objective sense: minimize
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    if verbose
        println("    MadNLP optimizer configured ($(round(time() - t_opt, digits=3))s)")
    end

    # initialize problem variables 
    t_vars = time()
    variables = set_variables!(optimizer, prob.trajectory)
    if verbose
        println("    variables set ($(round(time() - t_vars, digits=3))s)")
    end

    # # set callback function
    # if !isnothing(callback)
    #     MOI.set(optimizer, Ipopt.CallbackFunction(), callback(optimizer))
    # end

    # add linear constraints
    t_lincons = time()
    linear_constraints = AbstractLinearConstraint[filter(
        c->c isa AbstractLinearConstraint,
        prob.constraints,
    )...]
    Solvers.constrain!(
        optimizer,
        variables,
        linear_constraints,
        prob.trajectory;
        verbose = verbose,
    )
    if verbose
        println(
            "    linear constraints added: $(length(linear_constraints)) ($(round(time() - t_lincons, digits=3))s)",
        )
    end

    # set solver options
    set_options!(optimizer, options)

    if verbose
        println(
            "    optimizer initialization complete (total: $(round(time() - t_init_start, digits=3))s)",
        )
    end

    return optimizer, variables
end


function set_variables!(optimizer::AbstractOptimizer, traj::NamedTrajectory)
    # One optimizer variable per PACKED position: warp-free the historical
    # [datavec; global_data]; under a warp the derived timestep rows drop out
    # and the warp parameters (the duration variable) trail. (DirectTrajOpt#149)
    packed = collect(vec(traj))

    # add variables
    variables = MOI.add_variables(optimizer, length(packed))

    # set packed primal start (per-knot rows, globals, warp params)
    MOI.set(optimizer, MOI.VariablePrimalStart(), variables, packed)

    return variables
end

function update_trajectory!(
    prob::DirectTrajOptProblem,
    optimizer::AbstractOptimizer,
    variables::Vector{MOI.VariableIndex},
)
    z = MOI.get(optimizer, MOI.VariablePrimal(), variables)
    if prob.trajectory.warp !== nothing
        # Phase 1b (DTO#149): under a warp the packed vector goes through
        # unpack! — update! with a packed vector throws (the derived timestep
        # rows are never decision data).
        unpack!(prob.trajectory, z)
    else
        update!(prob.trajectory, z, type = :both)
    end
    return nothing
end


# ----------------------------------------------------------------------------
# Optimizer Configuration/Options
# ----------------------------------------------------------------------------


"""
    _MadNLPCallbackAdapter(inner)

Wrap a `DirectTrajOpt.AbstractIntermediateCallback` so MadNLP can call it
with its native `(solver, mode)` signature.

The adapter:
- **Filters mode to `UserCallbackRegular`.** MadNLP also invokes user
  callbacks during feasibility restoration and robust mode; those phases
  surface intermediate IPM state that's typically not meaningful to a
  trajectory-level callback, so they're silently skipped (return `true`).
  This makes `solver.cnt.k` monotonic from the callback's point of view.
- **Translates `solver.x` → `MadNLP.variable(solver.x)`** to strip the
  slack tail and hand back just the NLP primal.
- **Forwards `solver.cnt.k`** as the iteration index.
"""
struct _MadNLPCallbackAdapter <: MadNLP.AbstractUserCallback
    inner::DirectTrajOpt.AbstractIntermediateCallback
end

function (a::_MadNLPCallbackAdapter)(
    solver::MadNLP.AbstractMadNLPSolver,
    mode::MadNLP.AbstractUserCallbackStatus,
)
    mode isa MadNLP.UserCallbackRegular || return true
    return a.inner(MadNLP.variable(solver.x), solver.cnt.k)
end

function DirectTrajOpt.set_options!(optimizer::AbstractOptimizer, options::MadNLPOptions)
    ignored_options = [:eval_hessian]

    # Auto-couple: an AbstractIntermediateCallback needs the full primal vector,
    # which requires fixed_variable_treatment = RelaxBound. We only override when
    # MadNLP's own conditional default (`kkt_system <: SparseCondensedKKTSystem ?
    # RelaxBound : MakeParameter`) would otherwise pick `MakeParameter` and break
    # the callback. When the user has selected a kkt_system whose default is
    # already `RelaxBound`, MadNLP's if-one-liner gets to do its job untouched.
    # Raw MadNLP callbacks are presumed to manage this themselves.
    if options.intermediate_callback isa DirectTrajOpt.AbstractIntermediateCallback &&
       options.fixed_variable_treatment === nothing
        madnlp_default_is_relax_bound =
            options.kkt_system isa Type &&
            options.kkt_system <: MadNLP.SparseCondensedKKTSystem
        if !madnlp_default_is_relax_bound
            @info "Setting fixed_variable_treatment = MadNLP.RelaxBound for AbstractIntermediateCallback (MadNLP's kkt_system default would otherwise eliminate fixed vars from solver.x)"
            optimizer.options[:fixed_variable_treatment] = MadNLP.RelaxBound
        end
    end

    for name in fieldnames(typeof(options))
        value = getfield(options, name)
        if name in ignored_options
            continue
        end
        # `nothing` means "use MadNLP's own default" — don't overwrite the optimizer's
        # internal dict in that case. Applies to the pass-through fields
        # (linear_solver, array_type, kkt_system, cudss_ordering).
        if value === nothing
            continue
        end
        if name == :print_level
            optimizer.options[name] = MadNLP.LogLevels(value)
        elseif name == :hessian_approximation
            hessian_approximation = MadNLP.ExactHessian
            hessian_approximation =
                ((value == "compact_lbfgs") ? MadNLP.CompactLBFGS : hessian_approximation)
            optimizer.options[name] = hessian_approximation
        elseif name == :intermediate_callback
            if value isa DirectTrajOpt.AbstractIntermediateCallback
                # Wrap solver-agnostic callbacks in the MadNLP-shaped adapter.
                optimizer.options[name] = _MadNLPCallbackAdapter(value)
            elseif value isa MadNLP.AbstractUserCallback
                # Raw MadNLP callbacks pass through unwrapped.
                optimizer.options[name] = value
            else
                throw(
                    ArgumentError(
                        "intermediate_callback must be a subtype of " *
                        "`DirectTrajOpt.AbstractIntermediateCallback` or " *
                        "`MadNLP.AbstractUserCallback`, got $(typeof(value))",
                    ),
                )
            end
        else
            optimizer.options[name] = value
        end
    end
    return nothing
end


# ----------------------------------------------------------------------------
# Optimizer Tests
# ----------------------------------------------------------------------------


@testitem "testing MadNLP.jl solver" begin

    # include("../../test/test_utils.jl")
    include("../../test/madnlp_test_utils.jl")

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

    solve!(prob; options = MadNLPOptions(max_iter = 100))
end

@testitem "testing MadNLP.jl solver with NonlinearGlobalKnotPointConstraint" begin

    # include("../../test/test_utils.jl")
    include("../../test/madnlp_test_utils.jl")

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

    solve!(prob; options = MadNLPOptions(max_iter = 100))

    # Verify constraint is satisfied at each timestep
    for k = 2:(traj.N-1)
        u = traj[k][:u]
        g = traj.global_data[traj.global_components[:g]]
        @test norm(u) * (1.0 + norm(g)) <= 2.0 + 1e-6
    end
end

@testitem "testing solution trajectory independent of choice of solver" begin

    # include("../../test/test_utils.jl)
    # include("../../test/madnlp_test_utils.jl")
    include("../../test/solver_test_utils.jl")

    seed = rand(UInt64)

    prob_ipopt = get_seeded_prob_solved(seed, IpoptSolverExt.IpoptOptions(; max_iter = 100))
    prob_madnlp = get_seeded_prob_solved(seed, MadNLPOptions(; max_iter = 100))

    traj_ipopt = prob_ipopt.trajectory
    traj_madnlp = prob_madnlp.trajectory

    traj_dist = (traj_madnlp.data[:, :] .- traj_ipopt.data[:, :]) .^ 2
    traj_dist = sqrt(sum(traj_dist)) / length(traj_dist)

    @test traj_dist < 1e-4
end

# ----------------------------------------------------------------------------
# SolveStats contract on the MadNLP arm (C2 — spec-20260830-madnlp-first-flip):
# populated on the success path AND both failure paths (status-based limit and
# thrown), with the Ipopt arm's return convention.
# ----------------------------------------------------------------------------

@testitem "solve! returns populated SolveStats on the MadNLP arm (success path)" setup=[
    DTOTestHelpers,
] begin
    using DirectTrajOpt.Solvers: SolveStats, solve_status_symbol

    prob, _ = make_standard_prob()
    stats = solve!(prob; options = MadNLPOptions(max_iter = 100), verbose = false)

    @test stats isa SolveStats
    @test stats.solver === :madnlp
    @test stats.status ∈ (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
    @test solve_status_symbol(stats.status) === :solved
    # iterations come from MadNLP's model counter (cnt.k, via MOI.BarrierIterations)
    @test stats.iterations >= 1
    @test isfinite(stats.objective_value)
    @test stats.solve_time_s >= 0
    @test !isempty(stats.raw_status)
end

@testitem "solve! returns failure-class SolveStats when the iteration limit binds (MadNLP)" setup=[
    DTOTestHelpers,
] begin
    using DirectTrajOpt.Solvers: SolveStats, solve_status_symbol

    prob, _ = make_standard_prob()
    stats = solve!(prob; options = MadNLPOptions(max_iter = 1), verbose = false)

    @test stats isa SolveStats
    @test stats.solver === :madnlp
    @test stats.status == MOI.ITERATION_LIMIT
    @test solve_status_symbol(stats.status) === :limit
    @test stats.iterations == 1
    @test isfinite(stats.objective_value)
    @test stats.solve_time_s >= 0
    @test !isempty(stats.raw_status)
end

@testitem "solve! returns populated failure SolveStats when the solve throws (MadNLP)" setup=[
    DTOTestHelpers,
] begin
    using DirectTrajOpt.Solvers: SolveStats, solve_status_symbol

    mutable struct _ThrowingCallback <: DirectTrajOpt.AbstractIntermediateCallback
        fired::Base.RefValue{Bool}
    end
    function (cb::_ThrowingCallback)(::AbstractVector, ::Integer)
        cb.fired[] = true
        error("callback exploded mid-solve")
    end

    cb = _ThrowingCallback(Ref(false))
    prob, _ = make_standard_prob()
    # MadNLP's default rethrow_error=true surfaces the callback error; solve!
    # must convert it into populated stats instead of propagating.
    stats = solve!(
        prob;
        options = MadNLPOptions(max_iter = 100, intermediate_callback = cb),
        verbose = false,
    )

    @test cb.fired[]                       # the throw really happened inside MadNLP
    @test stats isa SolveStats             # failure became stats, not a thrown-only path
    @test stats.solver === :madnlp
    @test stats.status == MOI.OTHER_ERROR  # MadNLP INTERNAL_ERROR → MOI fallback
    @test solve_status_symbol(stats.status) === :error
    @test occursin("callback exploded", stats.raw_status)
    @test stats.solve_time_s >= 0
end

# ----------------------------------------------------------------------------
# Loud kwarg application (zero silent drops — the ipopt_solver/solver.jl:18
# convention, mirrored on the MadNLP arm).
# ----------------------------------------------------------------------------

@testitem "MadNLP solve! warns loudly on unknown kwargs" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    @test_logs (:warn, r"Unknown solver option: totally_fake_option") match_mode = :any begin
        solve!(
            prob;
            options = MadNLPOptions(max_iter = 5),
            verbose = false,
            totally_fake_option = 42,
        )
    end
end

@testitem "MadNLP solve! does not warn for mapped-with-translation kwargs (eval_hessian)" setup=[
    DTOTestHelpers,
] begin
    using Logging
    prob, _ = make_standard_prob()
    logs, _ = Test.collect_test_logs() do
        solve!(
            prob;
            options = MadNLPOptions(max_iter = 5),
            verbose = false,
            eval_hessian = false,
        )
    end
    @test !any(
        l -> l.level == Logging.Warn && occursin("Unknown solver option", l.message),
        logs,
    )
end
