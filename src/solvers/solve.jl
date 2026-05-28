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
function DirectTrajOpt._solve(prob::DirectTrajOptProblem, options::Any; kwargs...)
    @error "Invalid options argument (an instance of $(typeof(options)), not a subtype of $(AbstractSolverOptions))"
    return nothing
end

"""
function _solve_with_kwargs(
    prob::DirectTrajOptProblem,
    options::Any;
    kwargs...
)

Stub
"""
function DirectTrajOpt._solve_with_kwargs(
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
function DirectTrajOpt._solve(
    prob::DirectTrajOptProblem,
    options::AbstractSolverOptions;
    kwargs...,
)
    @error "No solver backend with matching options argument type (an instance of $(typeof(options)))"
    return nothing
end

"""
function _solve_with_kwargs(
    prob::DirectTrajOptProblem,
    options::AbstractSolverOptions;
    kwargs...
)

Specialized interface for adding support for a new solver in DirectTrajOpt.jl

    # Arguments
    - `prob::DirectTrajOptProblem`: The trajectory optimization problem to solve.
    - `options::AbstractSolverOptions`: The solver-specific options struct accompanying the problem to solve
    - `kwargs...`: Additional keyword arguments

    # Returns
    - `Any`: The solver interface may return `nothing`, or else it may return statistics pertaining to the success/failure of the solve
"""
function DirectTrajOpt._solve_with_kwargs(
    prob::DirectTrajOptProblem,
    options::AbstractSolverOptions;
    kwargs...,
)
    @error "No solver backend with matching options argument type (an instance of $(typeof(options)))"
    return nothing
end


"""
    _default_save_path() -> String

Build a default save path of the form `/tmp/pulse_<random>.jld2`. Used by
`solve!` when the user enables `save_solution` without naming a path.
"""
function _default_save_path()
    return joinpath("/tmp", "pulse_" * Random.randstring(12) * ".jld2")
end

"""
    _write_solution(path, prob, best_callback)

Serialize the post-solve `prob.trajectory` (and, if available, the best
intermediate trajectory tracked by `best_callback`) to `path` as a JLD2
file. Best-effort: failures (e.g. unwritable directory) are reported as
warnings without aborting the solve.
"""
function _write_solution(
    path::AbstractString,
    prob::DirectTrajOptProblem,
    best_callback;
    verbose::Bool = false,
)
    try
        mkpath(dirname(path))
        payload = Dict{String,Any}("trajectory" => prob.trajectory)
        if best_callback isa BestPulseCallback
            payload["best_trajectory"] = best_trajectory(best_callback)
            payload["best_objective"] = best_objective(best_callback)
            payload["best_iteration"] = best_iteration(best_callback)
            payload["best_primal_infeasibility"] = best_primal_infeasibility(best_callback)
        end
        JLD2.jldsave(String(path); payload...)
        if verbose
            @info "DirectTrajOpt: saved solution to $path"
        end
    catch err
        @warn "DirectTrajOpt: failed to write solution JLD2" path = path exception = err
    end
    return nothing
end

"""
    solve!(
        prob::DirectTrajOptProblem;
        options=<default>,
        verbose=true,
        callback=nothing,
        track_best=true,
        inf_pr_threshold=1e-4,
        save_solution=true,
        save_path=nothing,
        kwargs...
    )

Solve a `DirectTrajOptProblem` and return the installed
`BestPulseCallback` (or `nothing` if `track_best=false`).

# Callback handling

`callback` accepts a solver-specific `Function` (Ipopt only), an
`AbstractIntermediateCallback` (works for both Ipopt and MadNLP), or
`nothing`. When `track_best=true`, a `BestPulseCallback` is installed in
addition to whatever the user passes; if the user already passes an
`AbstractIntermediateCallback`, the two are combined via
`CompositeIntermediateCallback`.

# Best-pulse tracking

With `track_best=true` (default), the solver records the iterate with
the smallest objective value whose primal infeasibility is at or below
`inf_pr_threshold` (default `1e-4`). After the solve returns, the saved
trajectory snapshot is accessible via `best_trajectory(cb)`.

# JLD2 dump

With `save_solution=true` (default), the final trajectory — and, if
tracked, the best intermediate trajectory — are dumped to a JLD2 file.
`save_path` overrides the default location; when left at `nothing`, the
file lands in `/tmp/pulse_<random>.jld2`.

# Backwards-compatible behavior

The problem's trajectory is still updated in place. Existing callers
that ignore the return value continue to work unchanged.
"""
function DirectTrajOpt.solve!(
    prob::DirectTrajOptProblem;
    options = (Solvers._DefaultSolverOptions[])(),
    verbose::Bool = true,
    callback = nothing,
    track_best::Bool = true,
    inf_pr_threshold::Real = 1e-4,
    save_solution::Bool = true,
    save_path::Union{Nothing,AbstractString} = nothing,
    kwargs...,
)
    best_callback =
        track_best ? BestPulseCallback(prob; inf_pr_threshold = inf_pr_threshold) :
        nothing

    intermediate_cb = if best_callback !== nothing &&
                         callback isa AbstractIntermediateCallback
        CompositeIntermediateCallback(
            AbstractIntermediateCallback[best_callback, callback],
        )
    elseif best_callback !== nothing
        best_callback
    elseif callback isa AbstractIntermediateCallback
        callback
    else
        nothing
    end

    # User-supplied Function callbacks are Ipopt-specific (factory form).
    fn_callback = callback isa Function ? callback : nothing

    DirectTrajOpt._solve(
        prob,
        options;
        verbose = verbose,
        callback = fn_callback,
        intermediate_callback = intermediate_cb,
        kwargs...,
    )

    if save_solution
        path = save_path === nothing ? _default_save_path() : String(save_path)
        _write_solution(path, prob, best_callback; verbose = verbose)
    end

    return best_callback
end


# Coverage targets: src/solvers/solve.jl (35% → ~80%)

@testitem "get_num_variables without globals" setup=[DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory()
    integrators = [BilinearIntegrator(G, :x, :u, traj)]
    J = QuadraticRegularizer(:u, traj, 1.0)
    prob = DirectTrajOptProblem(traj, J, integrators)

    @test traj.global_dim == 0
    @test Solvers.get_num_variables(prob) == traj.dim * traj.N
end

@testitem "get_num_variables with globals" setup=[DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [BilinearIntegrator(G, :x, :u, traj)]
    J = QuadraticRegularizer(:u, traj, 1.0)
    prob = DirectTrajOptProblem(traj, J, integrators)

    @test traj.global_dim > 0
    @test Solvers.get_num_variables(prob) == traj.dim * traj.N + traj.global_dim
end

@testitem "get_nonlinear_constraints — dynamics only" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    # Remove the nonlinear constraint to test dynamics-only path
    G, traj = bilinear_dynamics_and_trajectory()
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = QuadraticRegularizer(:u, traj, 1.0)
    prob_dyn = DirectTrajOptProblem(traj, J, integrators)
    nl_cons = Solvers.get_nonlinear_constraints(prob_dyn)

    # All dynamics → equality bounds (0, 0)
    @test all(c -> c.lower == 0.0 && c.upper == 0.0, nl_cons)
end

@testitem "get_nonlinear_constraints — inequality" setup=[DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory()
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = QuadraticRegularizer(:u, traj, 1.0)
    g_ineq = NonlinearKnotPointConstraint(
        u -> [norm(u) - 1.0],
        :u,
        traj;
        times = 2:(traj.N-1),
        equality = false,
    )
    prob =
        DirectTrajOptProblem(traj, J, integrators; constraints = AbstractConstraint[g_ineq])
    nl_cons = Solvers.get_nonlinear_constraints(prob)

    dynamics_dim = sum(i.dim for i in integrators)
    ineq_cons = nl_cons[(dynamics_dim+1):end]
    @test all(c -> c.lower == -Inf && c.upper == 0.0, ineq_cons)
end

@testitem "get_nonlinear_constraints — equality" setup=[DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory()
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = QuadraticRegularizer(:u, traj, 1.0)
    g_eq = NonlinearKnotPointConstraint(
        u -> [norm(u) - 0.5],
        :u,
        traj;
        times = 2:(traj.N-1),
        equality = true,
    )
    prob =
        DirectTrajOptProblem(traj, J, integrators; constraints = AbstractConstraint[g_eq])
    nl_cons = Solvers.get_nonlinear_constraints(prob)

    dynamics_dim = sum(i.dim for i in integrators)
    eq_cons = nl_cons[(dynamics_dim+1):end]
    @test all(c -> c.lower == 0.0 && c.upper == 0.0, eq_cons)
end

@testitem "_solve error stubs" setup=[DTOTestHelpers] begin
    G, traj = bilinear_dynamics_and_trajectory()
    integrators = [BilinearIntegrator(G, :x, :u, traj)]
    J = QuadraticRegularizer(:u, traj, 1.0)
    prob = DirectTrajOptProblem(traj, J, integrators)

    # Ensure CI knows these are meant to error; maybe wrap in a try block?

    # Non-AbstractSolverOptions type
    @test DirectTrajOpt._solve(prob, "not_options") === nothing

    # AbstractSolverOptions subtype with no backend
    struct _FakeSolverOptions <: Solvers.AbstractSolverOptions end
    @test DirectTrajOpt._solve(prob, _FakeSolverOptions()) === nothing
end

@testitem "solve! default dispatch uses Ipopt" setup=[DTOTestHelpers] begin
    prob, _ = make_standard_prob()
    @test Solvers._get_DefaultSolverOptions() == IpoptSolverExt.IpoptOptions
    traj_before = deepcopy(prob.trajectory.data)
    solve!(prob; max_iter = 2, print_level = 0, verbose = false)
    @test prob.trajectory.data != traj_before
end
