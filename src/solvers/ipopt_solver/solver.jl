using NamedTrajectories
using DirectTrajOpt
using MathOptInterface
const MOI = MathOptInterface
using Ipopt
using TestItemRunner
using Libdl  # Added for Pardiso library loading

export solve!

export solve!

"""
    solve!(
        prob::DirectTrajOptProblem;
        options::IpoptOptions=IpoptOptions(),
        max_iter::Int=options.max_iter,
        verbose::Bool=true,
        linear_solver::String=options.linear_solver,
        print_level::Int=options.print_level,
        callback=nothing
    )

Solve a direct trajectory optimization problem using Ipopt.

# Arguments
- `prob::DirectTrajOptProblem`: The trajectory optimization problem to solve.
- `options::IpoptOptions`: Ipopt solver options. Default is `IpoptOptions()`.
- `max_iter::Int`: Maximum number of iterations for the optimization solver.
- `verbose::Bool`: If `true`, print solver progress information.
- `linear_solver::String`: Linear solver to use (e.g., "mumps", "pardiso", "ma27", "ma57", "ma77", "ma86", "ma97").
- `print_level::Int`: Ipopt print level (0-12). Higher values provide more detailed output.
- `callback::Function`: Optional callback function to execute during optimization.

# Returns
- `nothing`: The problem's trajectory is updated in place with the optimized solution.

# Example
```julia
prob = DirectTrajOptProblem(trajectory, objective, dynamics)
solve!(prob; max_iter=100, verbose=true)
```
"""
function DTO.Solvers.solve!(
    prob::DirectTrajOptProblem;
    options::IpoptOptions=IpoptOptions(),
    max_iter::Int=options.max_iter,
    verbose::Bool=true,
    linear_solver::String=options.linear_solver,
    print_level::Int=options.print_level,
    callback=nothing
)
    options.max_iter = max_iter
    options.linear_solver = linear_solver
    options.print_level = print_level

    optimizer, variables = get_optimizer_and_variables(prob, options, callback, verbose=verbose)

    MOI.optimize!(optimizer)

    update_trajectory!(prob, optimizer, variables)

    return nothing
end

# TODO: take another look at this
function remove_slack_variables!(prob::DirectTrajOptProblem)

    slack_var_names = Symbol[]

    for con ∈ prob.constraints
        if con isa L1SlackConstraint
            append!(slack_var_names, con.slack_names)
        end
    end

    prob.trajectory = remove_components(prob.trajectory, slack_var_names)
    return nothing
end

function get_num_variables(prob::DirectTrajOptProblem)
    n_vars = prob.trajectory.dim * prob.trajectory.N

    for global_vars_i ∈ values(prob.trajectory.global_data)
        n_global_vars = length(global_vars_i)
        n_vars += n_global_vars
    end

    return n_vars
end

function get_nonlinear_constraints(prob)
    # Compute dynamics dimension from integrators (same as TrajectoryDynamics does)
    dynamics_dim = 0
    # TODO: this is hacky as time integrator is being checked for, which should really bea linear constraint
    for integrator in prob.integrators
        # Get the state dimension from the trajectory using the integrator's x_name, x_names, or t_name
        if hasfield(typeof(integrator), :x_name)
            dynamics_dim += prob.trajectory.dims[integrator.x_name]
        elseif hasfield(typeof(integrator), :x_names)
            for x_name in integrator.x_names
                dynamics_dim += prob.trajectory.dims[x_name]
            end
        elseif hasfield(typeof(integrator), :t_name)
            dynamics_dim += prob.trajectory.dims[integrator.t_name]
        else
            error("Integrator type $(typeof(integrator)) must have either x_name, x_names, or t_name field")
        end
    end
    n_dynamics_constraints = dynamics_dim * (prob.trajectory.N - 1)

    nl_cons = fill(MOI.NLPBoundsPair(0.0, 0.0), n_dynamics_constraints)

    for nl_con ∈ filter(c -> c isa AbstractNonlinearConstraint, prob.constraints)
        if nl_con.equality
            append!(nl_cons, fill(MOI.NLPBoundsPair(0.0, 0.0), nl_con.dim))
        else
            append!(nl_cons, fill(MOI.NLPBoundsPair(-Inf, 0.0), nl_con.dim))
        end
    end

    return nl_cons
end

function get_optimizer_and_variables(
    prob::DirectTrajOptProblem, 
    options::IpoptOptions,
    callback::Union{Nothing, Function};
    verbose::Bool=true
)
    t_init_start = time()
    if verbose
        println("    initializing optimizer...")
    end

    # get evaluator
    t_eval = time()
    evaluator = IpoptEvaluator(prob; eval_hessian=options.eval_hessian, verbose=verbose)
    if verbose
        println("    evaluator created ($(round(time() - t_eval, digits=3))s)")
    end

    # get the MOI specific nonlinear constraints
    t_nlcons = time()
    nl_cons = get_nonlinear_constraints(prob)
    if verbose
        println("    NL constraint bounds extracted ($(round(time() - t_nlcons, digits=3))s)")
    end

    # build NLP block data
    t_block = time()
    block_data = MOI.NLPBlockData(nl_cons, evaluator, true)
    if verbose
        println("    NLP block data built ($(round(time() - t_block, digits=3))s)")
    end

    # initialize optimizer 
    t_opt = time()
    optimizer = Ipopt.Optimizer()

    # set NLP block data
    MOI.set(optimizer, MOI.NLPBlock(), block_data)

    # set objective sense: minimize
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    if verbose
        println("    Ipopt optimizer configured ($(round(time() - t_opt, digits=3))s)")
    end

    # initialize problem variables 
    t_vars = time()
    variables = set_variables!(optimizer, prob.trajectory)
    if verbose
        println("    variables set ($(round(time() - t_vars, digits=3))s)")
    end

    # set callback function
    if !isnothing(callback)
        MOI.set(optimizer, Ipopt.CallbackFunction(), callback(optimizer))
    end

    # add linear constraints
    t_lincons = time()
    linear_constraints = AbstractLinearConstraint[
        filter(c -> c isa AbstractLinearConstraint, prob.constraints)...
    ]
    constrain!(optimizer, variables, linear_constraints, prob.trajectory; verbose=verbose)
    if verbose
        println("    linear constraints added: $(length(linear_constraints)) ($(round(time() - t_lincons, digits=3))s)")
    end

    # set solver options
    set_options!(optimizer, options)

    if verbose
        println("    optimizer initialization complete (total: $(round(time() - t_init_start, digits=3))s)")
    end

    return optimizer, variables
end


# ----------------------------------------------------------------------------
#                         Optimizer Initialization
# ----------------------------------------------------------------------------

function set_variables!(
    optimizer::Ipopt.Optimizer,
    traj::NamedTrajectory
)
    data_dim = traj.dim * traj.N

    # add variables
    variables = MOI.add_variables(optimizer, data_dim + traj.global_dim)

    # set trajectory data
    MOI.set(
        optimizer,
        MOI.VariablePrimalStart(),
        variables[1:data_dim],
        collect(traj.datavec)
    )

    # set global data
    MOI.set(
        optimizer,
        MOI.VariablePrimalStart(),
        variables[data_dim .+ (1:traj.global_dim)],
        collect(traj.global_data)
    )

    return variables
end

function update_trajectory!(
    prob::DirectTrajOptProblem,
    optimizer::Ipopt.Optimizer, 
    variables::Vector{MOI.VariableIndex}
)
    update!(
        prob.trajectory, 
        MOI.get(optimizer, MOI.VariablePrimal(), variables),
        type=:both
    )
    return nothing
end


function set_options!(optimizer::Ipopt.Optimizer, options::IpoptOptions)
    ignored_options = [:eval_hessian, :refine]

    for name in fieldnames(typeof(options))
        value = getfield(options, name)
        if name in ignored_options
            continue
        end
        if name == :linear_solver
            if value == "pardiso"
                Libdl.dlopen("/usr/lib/x86_64-linux-gnu/liblapack.so.3", RTLD_GLOBAL)
                Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libomp.so.5", RTLD_GLOBAL)
            end
        end
        if !isnothing(value)
           optimizer.options[String(name)] = value
        end
    end
    return nothing
end

@testitem "testing solver" begin

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    g_u_norm = NonlinearKnotPointConstraint(u -> [norm(u) - 1.0], :u, traj; times=2:traj.N-1, equality=false)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints=AbstractConstraint[g_u_norm])

    solve!(prob; max_iter=100)
end

@testitem "testing solver with NonlinearGlobalKnotPointConstraint" begin

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global=true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)
    
    # Add global objective - minimize global parameter
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q=1.0)

    # Knot point constraint with global dependency
    # Couples control magnitude with global parameter
    g_ug = NonlinearGlobalKnotPointConstraint(
        ug -> begin
            u = ug[1:traj.dims[:u]]
            g = ug[traj.dims[:u] + 1:end]
            return [norm(u) * (1.0 + norm(g)) - 2.0]
        end,
        [:u], [:g], traj;
        times=2:traj.N-1,
        equality=false
    )

    prob = DirectTrajOptProblem(
        traj, J, integrators; 
        constraints=AbstractConstraint[g_ug]
    )

    solve!(prob; max_iter=100)
    
    # Verify constraint is satisfied at each timestep
    for k in 2:traj.N-1
        u = traj[k][:u]
        g = traj.global_data[traj.global_components[:g]]
        @test norm(u) * (1.0 + norm(g)) <= 2.0 + 1e-6
    end
end

@testitem "testing solver with NonlinearGlobalConstraint" begin

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global=true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)
    
    # Add global objective - minimize global parameter
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q=10.0)

    # Pure global constraint - bounds the global parameter
    g_global = NonlinearGlobalConstraint(
        g -> [norm(g) - 0.5],
        :g, traj;
        equality=false
    )

    prob = DirectTrajOptProblem(
        traj, J, integrators; 
        constraints=AbstractConstraint[g_global]
    )

    solve!(prob; max_iter=100)
    
    # Verify global variable is within constraint
    @test norm(traj.global_data[traj.global_components[:g]]) <= 0.5 + 1e-6
end

