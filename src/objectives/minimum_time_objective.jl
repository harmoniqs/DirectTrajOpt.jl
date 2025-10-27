export MinimumTimeObjective


"""
    MinimumTimeObjective

A type of objective that counts the time taken to complete a task.  `D` is a scaling factor.

"""
function MinimumTimeObjective(
    traj::NamedTrajectory;
    D::Float64=1.0
)
    Δt_index = traj.components[traj.timestep][1]
    Δt_indices = [index(t, Δt_index, traj.dim) for t = 1:traj.T-1]

    L = Z⃗::AbstractVector{<:Real} -> D * sum(Z⃗[Δt_indices])

    ∇L = (Z⃗::AbstractVector{<:Real}) -> begin
        ∇ = zeros(eltype(Z⃗), length(Z⃗))
        ∇[Δt_indices] .= D
        return ∇
    end

	∂²L = Z⃗::AbstractVector{<:Real} -> []
    ∂²L_structure = () -> []
    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

# Convenience constructor with D as positional argument
function MinimumTimeObjective(traj::NamedTrajectory, D::Real)
    return MinimumTimeObjective(traj; D=Float64(D))
end