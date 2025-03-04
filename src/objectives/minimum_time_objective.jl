export MinimumTimeObjective


"""
    MinimumTimeObjective

A type of objective that counts the time taken to complete a task.

Fields:
    `D`: a scaling factor
    `Δt_indices`: the indices of the time steps
    `eval_hessian`: whether to evaluate the Hessian

"""
function MinimumTimeObjective(
    traj::NamedTrajectory;
    D::Float64=1.0,
    timesteps_all_equal::Bool=false,
)
    @assert traj.timestep isa Symbol "Trajectory timestep must be a symbol, i.e. free time"
    if !timesteps_all_equal
        Δt_indices = [index(t, traj.components[traj.timestep][1], traj.dim) for t = 1:traj.T-1]

        L = Z⃗::AbstractVector -> D * sum(Z⃗[Δt_indices])

        ∇L = (Z⃗::AbstractVector) -> begin
            ∇ = zeros(eltype(Z⃗), length(Z⃗))
            ∇[Δt_indices] .= D
            return ∇
        end
    else
        Δt_T_index = index(traj.T, traj.components[traj.timestep][1], traj.dim)
        L = Z⃗::AbstractVector -> D * Z⃗[Δt_T_index]
        ∇L = (Z⃗::AbstractVector) -> begin
            ∇ = zeros(eltype(Z⃗), length(Z⃗))
            ∇[Δt_T_index] = D
            return ∇
        end

    end

	∂²L = Z⃗ -> []
    ∂²L_structure = () -> []
    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end