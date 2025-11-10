export MinimumTimeObjective

using TrajectoryIndexingUtils

"""
    MinimumTimeObjective <: AbstractObjective

Objective that minimizes total trajectory duration.

Computes:
```math
J = D \\sum_{k=1}^{N-1} \\Delta t_k
```

# Fields
- `D::Float64`: Scaling factor for minimum time objective

# Constructor
```julia
MinimumTimeObjective(traj::NamedTrajectory; D::Float64=1.0)
MinimumTimeObjective(traj::NamedTrajectory, D::Real)
```
"""
struct MinimumTimeObjective <: AbstractObjective
    D::Float64
end

function MinimumTimeObjective(
    traj::NamedTrajectory;
    D::Float64=1.0
)
    @assert traj.timestep isa Symbol "MinimumTimeObjective requires variable timestep"
    return MinimumTimeObjective(D)
end

# Convenience constructor with D as positional argument
function MinimumTimeObjective(traj::NamedTrajectory, D::Real)
    return MinimumTimeObjective(traj; D=Float64(D))
end

# Implement AbstractObjective interface

function objective_value(obj::MinimumTimeObjective, traj::NamedTrajectory)
    duration = 0.0
    for k in 1:traj.N-1
        duration += traj[k].timestep
    end
    return obj.D * duration
end

function gradient!(∇::AbstractVector, obj::MinimumTimeObjective, traj::NamedTrajectory)
    fill!(∇, 0.0)
    
    @assert traj.timestep isa Symbol "MinimumTimeObjective requires variable timestep"
    
    for k in 1:traj.N-1
        zₖ = traj[k]
        Δt_comps = zₖ.components[traj.timestep]
        Δt_indices = slice(k, Δt_comps, traj.dim)
        ∇[Δt_indices] .= obj.D
    end
    
    return nothing
end

function hessian_structure(obj::MinimumTimeObjective, traj::NamedTrajectory)
    # Linear objective has no Hessian
    return Tuple{Int,Int}[]
end

function hessian!(obj::MinimumTimeObjective, traj::NamedTrajectory)
    # Linear objective - Hessian is zero (nothing to compute)
    return nothing
end

function get_full_hessian(obj::MinimumTimeObjective, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    return spzeros(Z_dim, Z_dim)
end