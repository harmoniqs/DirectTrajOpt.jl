import MathOptInterface as MOI
import MadNLP

using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils

_get_module_by_name(name) =
    [mod for mod in reverse(Base.loaded_modules_order) if Symbol(mod) == Symbol(name)][1]

function _setup_madnlp(model::AbstractOptimizer; array_type = nothing, kwargs...)
    MadNLPMOI = _get_module_by_name(:MadNLPMOI)
    @assert MadNLPMOI isa Module "`MadNLP` extension `MadNLPMOI` not in `Base.loaded_modules_order`"

    #

    @assert model.solver isa Nothing "`model.solver` initialized prematurely"
    MadNLPMOI._setup_model(model)

    @assert model.nlp_model isa Nothing "`model.nlp_model` backend initialized erroneously"

    @assert model.needs_new_nlp "`model.nlp` backend initialized prematurely"
    MadNLPMOI._setup_nlp(model; array_type = array_type)

    @assert !model.needs_new_nlp "`model.nlp` backend failed to initialize"
    # model.options[:print_level] = (model.silent ? MadNLP.ERROR : model.options[:print_level])
    # model.options[:hessian_approximation] = (!model.hess_available ? MadNLP.CompactLBFGS : MadNLP.ExactHessian)
    # model.options[:jacobian_constant] = (model.has_only_linear_constraints ? true : false)

    for (_, s) in model.vector_nonlinear_oracle_constraints
        s.eval_f_timer = 0.0
        s.eval_jacobian_timer = 0.0
        s.eval_hessian_lagrangian_timer = 0.0
    end

    model.solver = MadNLP.MadNLPSolver(model.nlp; model.options..., kwargs...)

    return model
end

function _solve_madnlp(model::AbstractOptimizer; array_type = nothing)
    MadNLPMOI = _get_module_by_name(:MadNLPMOI)
    @assert MadNLPMOI isa Module "`MadNLP` extension `MadNLPMOI` not in `Base.loaded_modules_order`"

    #

    result = MadNLP.solve!(model.solver)
    model.result =
        (isa(array_type, Nothing) ? result : MadNLPMOI.copy_result_to_cpu(result))
    model.solve_time = model.solver.cnt.total_time
    model.solve_iterations = model.solver.cnt.k

    return model
end

function DirectTrajOpt._solve_with_kwargs(
    prob::DirectTrajOptProblem,
    options::MadNLPOptions;
    verbose::Bool = true,
    callback = nothing,
    array_type = nothing,
    kwargs...,
)
    # Apply kwargs to matching MadNLPOptions fields
    madnlp_fields = fieldnames(MadNLPOptions)
    madnlp_kwargs = Dict{Symbol,Any}()
    for (k, v) in kwargs
        if k in madnlp_fields
            setfield!(options, k, v)
        else
            # @warn "Unknown solver option: $k. Valid options: $(madnlp_fields)"
            push!(madnlp_kwargs, Pair(k, v))
        end
    end

    # Sync derived fields that depend on other fields.
    if haskey(madnlp_kwargs, :eval_hessian)
        # @warn "Manually specifying limited-memory option not yet implemented for MadNLP"
        setfield!(
            options,
            :hessian_approximation,
            pop!(madnlp_kwargs, :eval_hessian) ? "exact" : "compact_lbfgs",
        )
    end

    # Instantiate MadNLP.Optimizer <: MOI.AbstractOptimizer
    #   1. Set MOI.NLPBlock()
    #   2. Set MOI.ObjectiveSense()
    #   3. Set MOI.VariablePrimal()
    #   4. TODO: Set MOI.NLPBlockDualStart() (optional)
    #   5. TODO: Set callbacks (optional)
    #   6. Add linear constraints
    #   7. Set optimizer.options (involves conversions of the form convert(k::Symbol, v_in::Union{Real, String}, v_out::Any), where some of the v_out types are internal to MadNLP)

    optimizer, variables =
        get_optimizer_and_variables(prob, options, callback, verbose = verbose)

    # Calls MadNLPMOI._setup_model and MadNLPMOI._setup_nlp
    optimizer = _setup_madnlp(optimizer; array_type = array_type, madnlp_kwargs...)

    # Updates optimizer.result and calls MadNLPMOI.copy_result_to_cpu if necessary
    optimizer = _solve_madnlp(optimizer; array_type = array_type)

    # TODO: Verify this is working as expected
    update_trajectory!(prob, optimizer, variables)

    return nothing
end


@testitem "testing MadNLP.jl solver internals (using DirectTrajOpt._solve_with_kwargs with kkt_system=MadNLP.SparseUnreducedKKTSystem, linear_solver=MadNLP.LapackCPUSolver)" begin

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

    DirectTrajOpt._solve_with_kwargs(
        prob,
        MadNLPOptions(max_iter = 100);
        kkt_system = MadNLP.SparseUnreducedKKTSystem,
        linear_solver = MadNLP.LapackCPUSolver,
    )
end
