using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils


function remove_slack_variables!(prob::DirectTrajOptProblem)
    slack_var_names = Symbol[]

    for con ∈ prob.constraints
        if con isa L1SlackConstraint
            push!(slack_var_names, con.slack_name)
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
            error(
                "Integrator type $(typeof(integrator)) must have either x_name, x_names, or t_name field",
            )
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


"""
function _solve(
    prob::DirectTrajOptProblem,
    options::Any;
    kwargs...
)

Stub
"""
function _solve(
    prob::DirectTrajOptProblem,
    options::Any;
    kwargs...,
)
    @error "Invalid options argument (an instance of $(typeof(options)), not a subtype of $(AbstractSolverOptions))"
    return nothing
end

"""
function _solve(
    prob::DirectTrajOptProblem,
    options::AbstractSolverOptions;
    kwargs...
)

Generic interface for adding support for a new solver in DirectTrajOpt.jl

    # Arguments
    - `prob::DirectTrajOptProblem`: The trajectory optimization problem to solve.
    - `options::AbstractSolverOptions`: The solver-specific options struct accompanying the problem to solve
    - `kwargs...`: Additional keyword arguments

    # Returns
    - `Any`: The solver interface may return `nothing`, or else it may return statistics pertaining to the success/failure of the solve
"""
function _solve(
    prob::DirectTrajOptProblem,
    options::AbstractSolverOptions;
    kwargs...,
)
    @warn "No solver backend with matching options argument type (an instance of $(typeof(options)))"
    return nothing
end


"""
    solve!(
        prob::DirectTrajOptProblem;
        options::IpoptOptions=IpoptOptions(),
        verbose::Bool=true,
        callback=nothing,
        kwargs...
    )

Solve a direct trajectory optimization problem using Ipopt.

# Arguments
- `prob::DirectTrajOptProblem`: The trajectory optimization problem to solve.
- `options::IpoptOptions`: Ipopt solver options. Default is `IpoptOptions()`.
- `verbose::Bool`: If `true`, print solver progress information.
- `callback::Function`: Optional callback function to execute during optimization.
- `kwargs...`: Any field of `IpoptOptions` can be passed as a keyword argument. These
  override the corresponding field in `options`. See `IpoptOptions` for valid fields.

# Common keyword arguments
- `max_iter::Int`: Maximum solver iterations (default: 1000)
- `tol::Float64`: Convergence tolerance (default: 1e-8)
- `eval_hessian::Bool`: Use exact Hessians, or L-BFGS if false (default: true)
- `linear_solver::String`: Linear solver backend, e.g. `"mumps"`, `"pardiso"` (default: `"mumps"`)
- `print_level::Int`: Ipopt output verbosity 0–12 (default: 5)
- `mu_strategy::String`: Barrier parameter strategy (default: `"adaptive"`)

# Returns
- `nothing`: The problem's trajectory is updated in place with the optimized solution.

# Examples
```julia
# Simple usage
solve!(prob; max_iter=100, verbose=true)

# Override multiple Ipopt options
solve!(prob; max_iter=200, tol=1e-6, eval_hessian=false)

# Pass an options struct and override specific fields
solve!(prob; options=IpoptOptions(tol=1e-4), max_iter=500)
```
"""
function solve!(
    prob::DirectTrajOptProblem;
    options=(Solvers._DefaultSolverOptions[])(),
    verbose::Bool = true,
    callback = nothing,
    kwargs...,
)
    _solve(prob, options; verbose = verbose, callback = callback, kwargs...)

    return nothing
end
