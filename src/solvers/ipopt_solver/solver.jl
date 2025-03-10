using NamedTrajectories
using DirectTrajOpt
using MathOptInterface
const MOI = MathOptInterface
using Ipopt
using TestItemRunner

"""
   solve!(prob::DirectTrajOptProblem;
        init_traj=nothing,
        save_path=nothing,
        max_iter=prob.ipopt_options.max_iter,
        linear_solver=prob.ipopt_options.linear_solver,
        print_level=prob.ipopt_options.print_level,
        remove_slack_variables=false,
        callback=nothing
        # state_type=:unitary,
        # print_fidelity=false,
    )

    Call optimization solver to solve the quantum control problem with parameters and callbacks.

# Arguments
- `prob::DirectTrajOptProblem`: The quantum control problem to solve.
- `init_traj::NamedTrajectory`: Initial guess for the control trajectory. If not provided, a random guess will be generated.
- `save_path::String`: Path to save the problem after optimization.
- `max_iter::Int`: Maximum number of iterations for the optimization solver.
- `linear_solver::String`: Linear solver to use for the optimization solver (e.g., "mumps", "paradiso", etc).
- `print_level::Int`: Verbosity level for the solver.
- `callback::Function`: Callback function to call during optimization steps.
"""
function DC.solve!(
    prob::DirectTrajOptProblem;
    options::IpoptOptions=IpoptOptions(),
    max_iter::Int=options.max_iter,
    linear_solver::String=options.linear_solver,
    print_level::Int=options.print_level,
    callback=nothing
)
    options.max_iter = max_iter
    options.linear_solver = linear_solver
    options.print_level = print_level

    optimizer, variables = get_optimizer_and_variables(prob, options, callback)

    MOI.optimize!(optimizer)

    update_trajectory!(prob, optimizer, variables)

    remove_slack_variables!(prob)
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
end

function get_num_variables(prob::DirectTrajOptProblem)
    n_vars = prob.trajectory.dim * prob.trajectory.T

    for global_vars_i ∈ values(prob.trajectory.global_data)
        n_global_vars = length(global_vars_i)
        n_vars += n_global_vars
    end

    return n_vars
end

function get_nonlinear_constraints(prob)
    n_dynamics_constraints = prob.dynamics.dim * (prob.trajectory.T - 1)

    nl_cons = fill(MOI.NLPBoundsPair(0.0, 0.0), n_dynamics_constraints)

    for nl_con ∈ filter(c -> c isa AbstractNonlinearConstraint, prob.constraints)
        if nl_con.equality
            append!(nl_cons, fill(MOI.NLPBoundsPair(0.0, 0.0), nl_con.dim))
        else
            append!(nl_cons, fill(MOI.NLPBoundsPair(0.0, Inf), nl_con.dim))
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
    evaluator = IpoptEvaluator(prob; eval_hessian=options.eval_hessian)

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

    # set callback function
    if !isnothing(callback)
        MOI.set(optimizer, Ipopt.CallbackFunction(), callback)
    end

    # initialize problem variables 
    variables = set_variables!(optimizer, prob.trajectory)

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
    # initialize n variables with trajectory data
    n_traj_vars = traj.dim * traj.T
    n_vars = n_traj_vars + traj.global_dim


    # add variables
    variables = MOI.add_variables(optimizer, n_vars)

    # set trajectory data
    MOI.set(
        optimizer,
        MOI.VariablePrimalStart(),
        variables[1:n_traj_vars],
        collect(traj.datavec)
    )

    # set global variables
    running_vars = n_traj_vars
    for global_vars_i ∈ values(traj.global_data)
        n_global_vars = length(global_vars_i)
        MOI.set(
            optimizer,
            MOI.VariablePrimalStart(),
            variables[running_vars .+ (1:n_global_vars)],
            global_vars_i
        )
        running_vars += n_global_vars
    end

    return variables
end

function update_trajectory!(
    prob::DirectTrajOptProblem,
    optimizer::Ipopt.Optimizer, 
    variables::Vector{MOI.VariableIndex}
)

    n_vars = prob.trajectory.dim * prob.trajectory.T

    # get trajectory data
    datavec = MOI.get(
        optimizer,
        MOI.VariablePrimal(),
        variables[1:n_vars]
    )


    # get global variables after trajectory data
    global_keys = keys(prob.trajectory.global_data)
    global_values = []
    for global_var ∈ global_keys
        n_global_vars = length(prob.trajectory.global_data[global_var])
        push!(global_values, MOI.get(
            optimizer,
            MOI.VariablePrimal(),
            variables[n_vars .+ (1:n_global_vars)]
        ))
        n_vars += n_global_vars
    end
    global_data = (; (global_keys .=> global_values)...)

    prob.trajectory = NamedTrajectory(datavec, global_data, prob.trajectory)

    return nothing
end


function set_options!(optimizer::Ipopt.Optimizer, options::IpoptOptions)
    for name in fieldnames(typeof(options))
        value = getfield(options, name)
        if name == :eval_hessian
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
end

@testitem "testing solver" begin

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, traj, :x, :u),
        DerivativeIntegrator(traj, :u, :du),
        DerivativeIntegrator(traj, :du, :ddu)
    ]

    J = TerminalLoss(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    g_u_norm = NonlinearKnotPointConstraint(u -> [norm(u) - 1.0], :u, traj; times=2:traj.T-1, equality=false)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints=AbstractConstraint[g_u_norm])

    solve!(prob; max_iter=100)
end

 