module Problems

export DirectTrajOptProblem

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
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]
)
    traj_constraints = get_trajectory_constraints(traj)
    # Convert to AbstractConstraint vector to allow mixed types
    all_constraints = AbstractConstraint[constraints..., traj_constraints...]
    return DirectTrajOptProblem(traj, obj, integrators, all_constraints)
end

function DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::AbstractObjective,
    integrator::AbstractIntegrator;
    kwargs...
)
    return DirectTrajOptProblem(
        traj, 
        obj, 
        AbstractIntegrator[integrator];
        kwargs...
    )
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
        eq_con = EqualityConstraint(name, [1], val; label=con_label)
        push!(cons, eq_con)
    end

    # add final equality constraints
    for (name, val) ∈ pairs(traj.final)
        label = "final value of $name"
        eq_con = EqualityConstraint(name, [traj.N], val; label=label)
        push!(cons, eq_con)
    end
    
    # add bounds constraints
    for (name, bound) ∈ pairs(traj.bounds)
        if name ∈ keys(traj.initial) && name ∈ keys(traj.final) 
            ts = 2:traj.N-1
        elseif name ∈ keys(traj.initial) && !(name ∈ keys(traj.final))
            ts = 2:traj.N
        elseif name ∈ keys(traj.final) && !(name ∈ keys(traj.initial))
            ts = 1:traj.N-1
        else
            ts = 1:traj.N
        end
        con_label = "bounds on $name"
        bounds_con = BoundsConstraint(name, ts, bound; label=con_label)
        push!(cons, bounds_con)
    end
    
    # add time consistency constraint if trajectory has both :t and timestep variable
    timestep_var = traj.timestep
    if timestep_var isa Symbol && :t ∈ traj.names && timestep_var ∈ traj.names
        time_con = TimeConsistencyConstraint(; time_name=:t, timestep_name=timestep_var)
        push!(cons, time_con)
        
        # add t_1 = 0 constraint if not already specified in initial
        if :t ∉ keys(traj.initial)
            t_init_con = EqualityConstraint(:t, [1], [0.0]; label="initial time t₁ = 0")
            push!(cons, t_init_con)
        end
    end

    return cons
end

function Base.show(io::IO, prob::DirectTrajOptProblem)
    println(io, "DirectTrajOptProblem")
    println(io, "   timesteps            = ", prob.trajectory.N)
    println(io, "   duration             = ", get_duration(prob.trajectory))
    println(io, "   variable names       = ", prob.trajectory.names)
    println(io, "   knot point dimension = ", prob.trajectory.dim)
end




end
