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
using JLD2

"""
    mutable struct DirectTrajOptProblem <: AbstractProblem

Stores all the information needed to set up and solve a DirectTrajOptProblem as well as the solution
after the solver terminates.

# Fields
- `optimizer::Ipopt.Optimizer`: Ipopt optimizer object
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
    trajectory_constraints(traj::NamedTrajectory)

Implements the initial and final value constraints and bounds constraints on the controls
and states as specified by traj.

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
        eq_con = EqualityConstraint(name, [traj.T], val, traj; label=label)
        push!(cons, eq_con)
    end
    # add bounds constraints
    for (name, bound) ∈ pairs(traj.bounds)
        if name ∈ keys(traj.initial) && name ∈ keys(traj.final) 
            ts = 2:traj.T-1
        elseif name ∈ keys(traj.initial) && !(name ∈ keys(traj.final))
            ts = 2:traj.T
        elseif name ∈ keys(traj.final) && !(name ∈ keys(traj.initial))
            ts = 1:traj.T-1
        else
            ts = 1:traj.T
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
    println(io, "   timesteps            = ", prob.trajectory.T)
    println(io, "   duration             = ", get_duration(prob.trajectory))
    println(io, "   variable names       = ", prob.trajectory.names)
    println(io, "   knot point dimension = ", prob.trajectory.dim)
end




end
