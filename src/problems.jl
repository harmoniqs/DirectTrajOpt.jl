module Problems

export DirectTrajOptProblem
export show_problem_details

export get_trajectory_constraints

using ..Objectives
using ..Integrators
using ..Constraints

using TrajectoryIndexingUtils
using NamedTrajectories
using TestItems
using LinearAlgebra

"""
    mutable struct DirectTrajOptProblem

A direct trajectory optimization problem containing all information needed for setup and solution.

# Fields
- `trajectory::NamedTrajectory`: The trajectory containing optimization variables and data
- `objective::AbstractObjective`: The objective function to minimize
- `integrators::Vector{<:AbstractIntegrator}`: The integrators defining system dynamics
- `constraints::Vector{<:AbstractConstraint}`: Constraints on the trajectory

# Constructors
```julia
DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::AbstractObjective,
    integrators::Vector{<:AbstractIntegrator};
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]
)
```

Create a problem from a trajectory, objective, and integrators. Trajectory constraints
(initial, final, bounds) are automatically extracted and added to the constraint list.
The dynamics object is created by the evaluator at solve time.

# Example
```julia
traj = NamedTrajectory((x = rand(2, 10), u = rand(1, 10)), timestep=:Δt)
obj = QuadraticRegularizer(:u, traj, 1.0)
integrator = BilinearIntegrator(G, :x, :u)
prob = DirectTrajOptProblem(traj, obj, integrator)
```
"""
mutable struct DirectTrajOptProblem
    trajectory::NamedTrajectory
    objective::AbstractObjective
    integrators::Vector{<:AbstractIntegrator}
    constraints::Vector{<:AbstractConstraint}
end

function DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::AbstractObjective,
    integrators::Vector{<:AbstractIntegrator};
    constraints::Vector{<:AbstractConstraint} = AbstractConstraint[],
)
    # Validate timestep bounds if trajectory has a timestep variable
    timestep_var = traj.timestep
    if timestep_var isa Symbol && !haskey(traj.bounds, timestep_var)
        @warn """
            Trajectory has timestep variable :$timestep_var but no bounds on it.
            Adding default lower bound of 0 to prevent negative timesteps.

            Recommended: Add explicit bounds when creating the trajectory:
              NamedTrajectory(...; Δt_bounds=(min, max))
            Example:
              NamedTrajectory(qtraj, N; Δt_bounds=(1e-3, 0.5))

            Or use timesteps_all_equal=true in problem options to fix timesteps.
            """ maxlog=1

        # Add lower bound of 0 to prevent negative timesteps
        # Create new trajectory with updated bounds
        timestep_dim = traj.dims[timestep_var]
        new_bounds = merge(traj.bounds, (; timestep_var => (zeros(timestep_dim), fill(Inf, timestep_dim))))
        
        # Extract component data
        comps_data = NamedTuple(name => traj[name] for name in traj.names)
        
        # Extract global component data if present
        if traj.global_dim > 0
            gcomps_data = NamedTuple(
                name => Vector(traj.global_data[traj.global_components[name]]) 
                for name in keys(traj.global_components)
            )
            traj = NamedTrajectory(
                comps_data,
                gcomps_data;
                timestep=traj.timestep,
                controls=traj.control_names,
                bounds=new_bounds,
                initial=traj.initial,
                final=traj.final,
                goal=traj.goal
            )
        else
            traj = NamedTrajectory(
                comps_data;
                timestep=traj.timestep,
                controls=traj.control_names,
                bounds=new_bounds,
                initial=traj.initial,
                final=traj.final,
                goal=traj.goal
            )
        end
    end

    traj_constraints = get_trajectory_constraints(traj)
    # Convert to AbstractConstraint vector to allow mixed types
    all_constraints = AbstractConstraint[constraints..., traj_constraints...]
    return DirectTrajOptProblem(traj, obj, integrators, all_constraints)
end

function DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::AbstractObjective,
    integrator::AbstractIntegrator;
    kwargs...,
)
    return DirectTrajOptProblem(traj, obj, AbstractIntegrator[integrator]; kwargs...)
end


"""
    get_trajectory_constraints(traj::NamedTrajectory)

Extract and create constraints from a NamedTrajectory's initial, final, and bounds specifications.

# Arguments
- `traj::NamedTrajectory`: Trajectory with specified initial conditions, final conditions, and/or bounds

# Returns
- `Vector{AbstractConstraint}`: Vector of constraints including:
  - Initial value equality constraints (from `traj.initial`)
  - Final value equality constraints (from `traj.final`)
  - Bounds constraints (from `traj.bounds`)

# Details
The function automatically handles time indices based on which constraints are specified:
- If both initial and final constraints exist for a component, bounds apply to interior points (2:N-1)
- If only initial exists, bounds apply from second point onward (2:N)
- If only final exists, bounds apply up to second-to-last point (1:N-1)
- If neither exist, bounds apply to all time points (1:N)
"""
function get_trajectory_constraints(traj::NamedTrajectory)

    cons = AbstractConstraint[]

    # add initial equality constraints
    for (name, val) ∈ pairs(traj.initial)
        con_label = "initial value of $name"
        eq_con = EqualityConstraint(name, [1], val; label = con_label)
        push!(cons, eq_con)
    end

    # add final equality constraints
    for (name, val) ∈ pairs(traj.final)
        label = "final value of $name"
        eq_con = EqualityConstraint(name, [traj.N], val; label = label)
        push!(cons, eq_con)
    end

    # add bounds constraints
    for (name, bound) ∈ pairs(traj.bounds)
        if name ∈ keys(traj.initial) && name ∈ keys(traj.final)
            ts = 2:(traj.N-1)
        elseif name ∈ keys(traj.initial) && !(name ∈ keys(traj.final))
            ts = 2:traj.N
        elseif name ∈ keys(traj.final) && !(name ∈ keys(traj.initial))
            ts = 1:(traj.N-1)
        else
            ts = 1:traj.N
        end
        con_label = "bounds on $name"
        bounds_con = BoundsConstraint(name, ts, bound; label = con_label)
        push!(cons, bounds_con)
    end

    # add time consistency constraint if trajectory has both :t and timestep variable
    timestep_var = traj.timestep
    if timestep_var isa Symbol && :t ∈ traj.names && timestep_var ∈ traj.names
        time_con = TimeConsistencyConstraint(; time_name = :t, timestep_name = timestep_var)
        push!(cons, time_con)

        # add t_1 = 0 constraint if not already specified in initial
        if :t ∉ keys(traj.initial)
            t_init_con = EqualityConstraint(:t, [1], [0.0]; label = "initial time t₁ = 0")
            push!(cons, t_init_con)
        end
    end

    return cons
end

"""
    show_problem_details(io::IO, prob::DirectTrajOptProblem)

Print the trajectory, objective, dynamics, and constraints sections of a problem.

This is used by both `DirectTrajOptProblem` and `QuantumControlProblem` display methods.
"""
function show_problem_details(io::IO, prob::DirectTrajOptProblem)
    traj = prob.trajectory

    # --- Trajectory section ---
    println(io, "  Trajectory")
    println(io, "    Timesteps: ", traj.N)
    println(io, "    Duration:  ", round(get_duration(traj), sigdigits = 6))
    println(io, "    Knot dim:  ", traj.dim)
    vars = join(["$n ($(traj.dims[n]))" for n in traj.names], ", ")
    println(io, "    Variables: ", vars)
    ctrl_str = isempty(traj.control_names) ? "(none)" : join(traj.control_names, ", ")
    println(io, "    Controls:  ", ctrl_str)
    if traj.global_dim > 0
        gvars = join(
            [
                "$n ($(length(traj.global_components[n])))" for
                n in keys(traj.global_components)
            ],
            ", ",
        )
        println(io, "    Globals:   ", gvars)
    end

    # --- Objective section ---
    obj = prob.objective
    if obj isa CompositeObjective
        n = length(obj.objectives)
        println(io, "  Objective ($n terms)")
        for (sub_obj, w) in zip(obj.objectives, obj.weights)
            w_str = string(round(w, sigdigits = 4))
            println(io, "    $(lpad(w_str, 8)) * ", sub_obj)
        end
    elseif obj isa NullObjective
        println(io, "  Objective: NullObjective")
    else
        println(io, "  Objective: ", obj)
    end

    # --- Dynamics section ---
    n_int = length(prob.integrators)
    println(io, "  Dynamics ($n_int integrators)")
    for integ in prob.integrators
        println(io, "    ", integ)
    end

    # --- Constraints section ---
    constraints = prob.constraints
    n_con = length(constraints)
    if n_con > 0
        n_eq = count(c -> c isa EqualityConstraint, constraints)
        n_bnd = count(c -> c isa BoundsConstraint, constraints)
        n_tc = count(c -> c isa TimeConsistencyConstraint, constraints)
        n_other = n_con - n_eq - n_bnd - n_tc

        parts = String[]
        n_eq > 0 && push!(parts, "$n_eq equality")
        n_bnd > 0 && push!(parts, "$n_bnd bounds")
        n_tc > 0 && push!(parts, "$n_tc time consistency")
        n_other > 0 && push!(parts, "$n_other other")

        println(io, "  Constraints ($n_con total: ", join(parts, ", "), ")")
        max_show = 10
        for (i, con) in enumerate(constraints)
            if i <= max_show
                if i < n_con
                    println(io, "    ", con)
                else
                    print(io, "    ", con)
                end
            elseif i == max_show + 1
                print(io, "    ... and $(n_con - max_show) more")
                break
            end
        end
    else
        print(io, "  Constraints: (none)")
    end
end

function Base.show(io::IO, prob::DirectTrajOptProblem)
    println(io, "DirectTrajOptProblem")
    show_problem_details(io, prob)
end

end
