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
    #   7. Set optimizer options (involves conversions of the form convert(k::Symbol, v_in::Union{Real, String}, v_out::Any), where some of the v_out types are internal to MadNLP)

    optimizer, variables =
        get_optimizer_and_variables(prob, options, callback, verbose = verbose)

    MOI.optimize!(optimizer)

    # TODO: Verify this is working as expected
    update_trajectory!(prob, optimizer, variables)

    return nothing
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
    data_dim = traj.dim * traj.N

    # add variables
    variables = MOI.add_variables(optimizer, data_dim + traj.global_dim)

    # set trajectory data
    MOI.set(
        optimizer,
        MOI.VariablePrimalStart(),
        variables[1:data_dim],
        collect(traj.datavec),
    )

    # set global data
    MOI.set(
        optimizer,
        MOI.VariablePrimalStart(),
        variables[data_dim .+ (1:traj.global_dim)],
        collect(traj.global_data),
    )

    return variables
end

function update_trajectory!(
    prob::DirectTrajOptProblem,
    optimizer::AbstractOptimizer,
    variables::Vector{MOI.VariableIndex},
)
    update!(
        prob.trajectory,
        MOI.get(optimizer, MOI.VariablePrimal(), variables),
        type = :both,
    )
    return nothing
end


# ----------------------------------------------------------------------------
# Optimizer Configuration/Options
# ----------------------------------------------------------------------------


function DirectTrajOpt.set_options!(optimizer::AbstractOptimizer, options::MadNLPOptions)
    ignored_options = [:eval_hessian]

    for name in fieldnames(typeof(options))
        value = getfield(options, name)
        if name in ignored_options
            continue
        end
        # TODO: allow internal defaults, i.e. do not set the internal options dict unless the user actually specified the associated opt
        if name == :print_level
            optimizer.options[name] = MadNLP.LogLevels(value)
        elseif name == :hessian_approximation
            hessian_approximation = MadNLP.ExactHessian
            hessian_approximation =
                ((value == "compact_lbfgs") ? MadNLP.CompactLBFGS : hessian_approximation)
            optimizer.options[name] = hessian_approximation
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
    prob_madnlp =
        get_seeded_prob_solved(seed, MadNLPOptions(; max_iter = 100))

    traj_ipopt = prob_ipopt.trajectory
    traj_madnlp = prob_madnlp.trajectory

    traj_dist = (traj_madnlp.data[:, :] .- traj_ipopt.data[:, :]) .^ 2
    traj_dist = sqrt(sum(traj_dist)) / length(traj_dist)

    @test traj_dist < 1e-4
end
