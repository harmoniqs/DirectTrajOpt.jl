export AllEqualConstraint
export TimeStepsAllEqualConstraint

struct AllEqualConstraint <: AbstractLinearConstraint
    indices::Vector{Int}
    bar_index::Int
    label::String
end

function TimeStepsAllEqualConstraint(
    traj::NamedTrajectory;
    label="timesteps all equal constraint"
)
    @assert traj.timestep isa Symbol
    indices = [index(k, traj.components[traj.timestep][1], traj.dim) for k ∈ 1:traj.N-1]
    bar_index = index(traj.N, traj.components[traj.timestep][1], traj.dim)
    return AllEqualConstraint(indices, bar_index, label)
end
