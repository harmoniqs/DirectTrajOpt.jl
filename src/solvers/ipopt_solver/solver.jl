using NamedTrajectories
using DirectTrajOpt
using MathOptInterface
const MOI = MathOptInterface
using Ipopt
using TestItemRunner

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
function DC.solve!(
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
        # Get the state dimension from the trajectory using the integrator's x_name or t_name
        if hasfield(typeof(integrator), :x_name)
            dynamics_dim += prob.trajectory.dims[integrator.x_name]
        elseif hasfield(typeof(integrator), :t_name)
            dynamics_dim += prob.trajectory.dims[integrator.t_name]
        else
            error("Integrator type $(typeof(integrator)) must have either x_name or t_name field")
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
    if verbose
        println("    initializing optimizer...")
    end

    # get evaluator
    evaluator = IpoptEvaluator(prob; eval_hessian=options.eval_hessian, verbose=verbose)

    # get the MOI specific nonlinear constraints
    nl_cons = get_nonlinear_constraints(prob)

    # build NLP block data
    block_data = MOI.NLPBlockData(nl_cons, evaluator, true)

    # initialize optimizer 
    optimizer = Ipopt.Optimizer()

    # set NLP block data
    MOI.set(optimizer, MOI.NLPBlock(), block_data)

    # set objective sense: minimize
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # initialize problem variables 
    variables = set_variables!(optimizer, prob.trajectory)

    # set callback function
    if !isnothing(callback)
        MOI.set(optimizer, Ipopt.CallbackFunction(), callback(optimizer))
    end

    # add linear constraints
    linear_constraints = AbstractLinearConstraint[
        filter(c -> c isa AbstractLinearConstraint, prob.constraints)...
    ]
    constrain!(optimizer, variables, linear_constraints; verbose=verbose)

    # set solver options
    set_options!(optimizer, options)

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
        BilinearIntegrator(G, traj, :x, :u),
        DerivativeIntegrator(traj, :u, :du),
        DerivativeIntegrator(traj, :du, :ddu)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    g_u_norm = NonlinearKnotPointConstraint(u -> [norm(u) - 1.0], :u, traj; times=2:traj.N-1, equality=false)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints=AbstractConstraint[g_u_norm])

    solve!(prob; max_iter=100)
end

