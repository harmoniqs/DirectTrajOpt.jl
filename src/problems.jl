module Problems

export DirectTrajOptProblem

export get_trajectory_constraints

using ..Objectives
using ..Integrators
using ..Dynamics
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
- `objective::Objective`: The objective function to minimize
- `dynamics::TrajectoryDynamics`: The system dynamics (integrators)
- `constraints::Vector{<:AbstractConstraint}`: Constraints on the trajectory

# Constructors
```julia
DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::Objective,
    integrators::Vector{<:AbstractIntegrator};
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]
)
```

Create a problem from a trajectory, objective, and integrators. Trajectory constraints
(initial, final, bounds) are automatically extracted and added to the constraint list.

# Example
```julia
traj = NamedTrajectory((x = rand(2, 10), u = rand(1, 10)), timestep=:Δt)
obj = QuadraticRegularizer(:u, traj, 1.0)
integrator = BilinearIntegrator(G, traj, :x, :u)
prob = DirectTrajOptProblem(traj, obj, integrator)
```
"""
mutable struct DirectTrajOptProblem
    trajectory::NamedTrajectory
    objective::Objective
    dynamics::TrajectoryDynamics
    constraints::Vector{<:AbstractConstraint}
end

function DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::Objective,
    integrators::Vector{<:AbstractIntegrator};
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]
)
    dynamics = TrajectoryDynamics(integrators, traj)
    traj_constraints = get_trajectory_constraints(traj)
    append!(constraints, traj_constraints)
    return DirectTrajOptProblem(traj, obj, dynamics, constraints)
end

function DirectTrajOptProblem(
    traj::NamedTrajectory,
    obj::Objective,
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
        eq_con = EqualityConstraint(name, [1], val, traj; label=con_label)
        push!(cons, eq_con)
    end

    # add final equality constraints
    for (name, val) ∈ pairs(traj.final)
        label = "final value of $name"
        eq_con = EqualityConstraint(name, [traj.N], val, traj; label=label)
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
        # bounds = collect(zip(bound[1], bound[2]))
        bounds_con = BoundsConstraint(name, ts, bound, traj; label=con_label)
        push!(cons, bounds_con)
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
