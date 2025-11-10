export GlobalObjective
export GlobalKnotPointObjective

using TrajectoryIndexingUtils

# ----------------------------------------------------------------------------- #
# GlobalObjective
# ----------------------------------------------------------------------------- #

"""
    GlobalObjective <: AbstractObjective

Objective that only involves global (non-time-varying) trajectory components.

Objective function ℓ operates on extracted global variable values:
```math
J = Q \\cdot \\ell(\\text{global\\_vars})
```

# Fields
- `ℓ::Function`: Objective function mapping global variables → scalar cost
- `global_names::Vector{Symbol}`: Names of global trajectory variables
- `Q::Float64`: Weight for the objective
- `∂²ℓ::Matrix{Float64}`: Preallocated local Hessian matrix (for global variables only)

# Constructor
```julia
GlobalObjective(
    ℓ::Function,
    global_names::Union{Symbol, AbstractVector{Symbol}},
    traj::NamedTrajectory;
    Q::Float64=1.0
)
```
"""
struct GlobalObjective <: AbstractObjective
    ℓ::Function
    global_names::Vector{Symbol}
    Q::Float64
    ∂²ℓ::Matrix{Float64}
end

function GlobalObjective(
    ℓ::Function,
    global_names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    Q::Float64=1.0
)
    # Preallocate local Hessian for global variables only
    n_global_vars = sum(length(traj.global_components[name]) for name in global_names)
    ∂²ℓ = zeros(n_global_vars, n_global_vars)
    return GlobalObjective(ℓ, Vector{Symbol}(global_names), Q, ∂²ℓ)
end

function GlobalObjective(ℓ::Function, global_name::Symbol, traj::NamedTrajectory; kwargs...)
    return GlobalObjective(ℓ, [global_name], traj; kwargs...)
end

# Implement AbstractObjective interface

function objective_value(obj::GlobalObjective, traj::NamedTrajectory)
    # Extract global components
    g_vals = vcat([traj.global_data[traj.global_components[name]] for name in obj.global_names]...)
    return obj.Q * obj.ℓ(g_vals)
end

function gradient!(∇::AbstractVector, obj::GlobalObjective, traj::NamedTrajectory)
    fill!(∇, 0.0)
    
    # Extract global components and their indices
    g_vals = vcat([traj.global_data[traj.global_components[name]] for name in obj.global_names]...)
    offset = traj.dim * traj.N
    global_indices = vcat([offset .+ traj.global_components[name] for name in obj.global_names]...)
    
    # Compute gradient using ForwardDiff
    ∇ℓ_local = ForwardDiff.gradient(obj.ℓ, g_vals)
    
    # Map to full gradient vector
    ∇[global_indices] .= obj.Q .* ∇ℓ_local
    
    return nothing
end

function hessian_structure(obj::GlobalObjective, traj::NamedTrajectory)
    # Dense Hessian - return upper triangle
    Z_dim = traj.dim * traj.N + traj.global_dim
    structure = Tuple{Int,Int}[]
    for i in 1:Z_dim
        for j in i:Z_dim
            push!(structure, (i, j))
        end
    end
    return structure
end

function hessian!(obj::GlobalObjective, traj::NamedTrajectory)
    # Extract global components
    g_vals = vcat([traj.global_data[traj.global_components[name]] for name in obj.global_names]...)
    
    # Compute Hessian in-place using ForwardDiff, with weight
    ForwardDiff.hessian!(
        obj.∂²ℓ,
        x -> obj.Q * obj.ℓ(x),
        g_vals
    )
    
    return nothing
end

function get_full_hessian(obj::GlobalObjective, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂²L = spzeros(Z_dim, Z_dim)
    
    # Get global indices
    offset = traj.dim * traj.N
    global_indices = vcat([offset .+ traj.global_components[name] for name in obj.global_names]...)
    
    # Map local Hessian to full matrix using simplified indexing
    ∂²L[global_indices, global_indices] .= obj.∂²ℓ
    
    return ∂²L
end

# ----------------------------------------------------------------------------- #
# Global KnotPointObjective
# ----------------------------------------------------------------------------- #

"""
    GlobalKnotPointObjective <: AbstractObjective

Knot point objective that includes both time-varying and global trajectory components.

Objective function ℓ operates on extracted variable values:
```math
J = \\sum_{k \\in \\text{times}} Q_k \\ell([x_k; g], p_k)
```

where ℓ receives both knot point variables and global variables concatenated.

# Fields
- `ℓ::Function`: Objective function mapping (knot_vars + global_vars, params) → scalar cost
- `var_names::Vector{Symbol}`: Names of trajectory variables at knot points
- `global_names::Vector{Symbol}`: Names of global trajectory variables
- `times::Vector{Int}`: Time indices where objective is evaluated
- `params::Vector`: Parameters for each time index
- `Qs::Vector{Float64}`: Weights for each time index
- `∂²ℓs::Vector{Matrix{Float64}}`: Preallocated local Hessians (one per timestep)
"""
struct GlobalKnotPointObjective <: AbstractObjective
    ℓ::Function
    var_names::Vector{Symbol}
    global_names::Vector{Symbol}
    times::Vector{Int}
    params::Vector
    Qs::Vector{Float64}
    ∂²ℓs::Vector{Matrix{Float64}}
end

function GlobalKnotPointObjective(
    ℓ::Function,
    names::AbstractVector{Symbol},
    global_names::AbstractVector{Symbol},
    traj::NamedTrajectory,
    params::AbstractVector;
    times::AbstractVector{Int}=1:traj.N,
    Qs::AbstractVector{Float64}=ones(length(times)),
)
    @assert length(Qs) == length(times) "Qs must have the same length as times"
    @assert length(params) == length(times) "params must have the same length as times"

    # Preallocate local Hessians (knot_vars + global_vars size)
    n_knot_vars = sum(length(traj.components[name]) for name in names)
    n_global_vars = sum(length(traj.global_components[name]) for name in global_names)
    n_total = n_knot_vars + n_global_vars
    
    ∂²ℓs = [zeros(n_total, n_total) for _ in times]

    return GlobalKnotPointObjective(
        ℓ,
        Vector{Symbol}(names),
        Vector{Symbol}(global_names),
        Vector{Int}(times),
        Vector(params),
        Vector{Float64}(Qs),
        ∂²ℓs
    )
end

function GlobalKnotPointObjective(
    ℓ::Function,
    names::AbstractVector{Symbol},
    global_names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    times::AbstractVector{Int}=1:traj.N,
    kwargs...
)
    params = [nothing for _ in times]
    ℓ_param = (x, _) -> ℓ(x)
    return GlobalKnotPointObjective(ℓ_param, names, global_names, traj, params; times=times, kwargs...)
end

# Implement AbstractObjective interface

function objective_value(obj::GlobalKnotPointObjective, traj::NamedTrajectory)
    J = 0.0
    for (i, t) in enumerate(obj.times)
        zₖ = traj[t]
        # Extract knot point variables
        x_vals = vcat([zₖ[name] for name in obj.var_names]...)
        # Extract global variables
        g_vals = vcat([traj.global_data[traj.global_components[name]] for name in obj.global_names]...)
        # Concatenate
        xg_vals = vcat(x_vals, g_vals)
        J += obj.Qs[i] * obj.ℓ(xg_vals, obj.params[i])
    end
    return J
end

function gradient!(∇::AbstractVector, obj::GlobalKnotPointObjective, traj::NamedTrajectory)
    fill!(∇, 0.0)
    
    # Pre-compute global indices
    global_offset = traj.dim * traj.N
    global_indices = vcat([global_offset .+ traj.global_components[name] for name in obj.global_names]...)
    
    for (i, t) in enumerate(obj.times)
        zₖ = traj[t]
        # Extract knot point variables and components
        x_vals = vcat([zₖ[name] for name in obj.var_names]...)
        x_comps = vcat([zₖ.components[name] for name in obj.var_names]...)
        # Extract global variables
        g_vals = vcat([traj.global_data[traj.global_components[name]] for name in obj.global_names]...)
        # Concatenate
        xg_vals = vcat(x_vals, g_vals)
        
        # Compute gradient using ForwardDiff
        ∇ℓ_local = ForwardDiff.gradient(
            xg -> obj.ℓ(xg, obj.params[i]),
            xg_vals
        )
        
        # Split gradient into knot point and global parts
        n_knot = length(x_vals)
        ∇ℓ_knot = ∇ℓ_local[1:n_knot]
        ∇ℓ_global = ∇ℓ_local[n_knot+1:end]
        
        # Map to full gradient vector
        knot_indices = slice(t, x_comps, traj.dim)
        ∇[knot_indices] .+= obj.Qs[i] .* ∇ℓ_knot
        ∇[global_indices] .+= obj.Qs[i] .* ∇ℓ_global
    end
    
    return nothing
end

function hessian_structure(obj::GlobalKnotPointObjective, traj::NamedTrajectory)
    # Dense Hessian - return upper triangle
    Z_dim = traj.dim * traj.N + traj.global_dim
    structure = Tuple{Int,Int}[]
    for i in 1:Z_dim
        for j in i:Z_dim
            push!(structure, (i, j))
        end
    end
    return structure
end

function hessian!(obj::GlobalKnotPointObjective, traj::NamedTrajectory)
    for (i, t) in enumerate(obj.times)
        zₖ = traj[t]
        # Extract knot point variables
        x_vals = vcat([zₖ[name] for name in obj.var_names]...)
        # Extract global variables
        g_vals = vcat([traj.global_data[traj.global_components[name]] for name in obj.global_names]...)
        # Concatenate
        xg_vals = vcat(x_vals, g_vals)
        
        # Compute local Hessian in-place using ForwardDiff, with weight
        ForwardDiff.hessian!(
            obj.∂²ℓs[i],
            xg -> obj.Qs[i] * obj.ℓ(xg, obj.params[i]),
            xg_vals
        )
    end
    
    return nothing
end

function get_full_hessian(obj::GlobalKnotPointObjective, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂²L = spzeros(Z_dim, Z_dim)
    
    # Pre-compute global indices
    global_offset = traj.dim * traj.N
    global_indices = vcat([global_offset .+ traj.global_components[name] for name in obj.global_names]...)
    
    for (i, t) in enumerate(obj.times)
        zₖ = traj[t]
        # Get knot point indices
        x_comps = vcat([zₖ.components[name] for name in obj.var_names]...)
        knot_indices = slice(t, x_comps, traj.dim)
        
        # All indices combined
        all_indices = vcat(knot_indices, global_indices)
        
        # Map local Hessian to full matrix using simplified indexing
        ∂²L[all_indices, all_indices] .+= obj.∂²ℓs[i]
    end
    
    return ∂²L
end

# ============================================================================ #

@testitem "testing GlobalObjective" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory(add_global=true)

    ℓ(g) = norm(g)^2  # Use quadratic for non-zero Hessian
    Q = 2.0

    OBJ = GlobalObjective(ℓ, :g, traj, Q=Q)

    test_objective(OBJ, traj)
end

@testitem "testing GlobalKnotPointObjective" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory(add_global=true)

    function ℓ(ug)
        u, g = ug[1:traj.dims[:u]], ug[traj.dims[:u] .+ 1:end]
        return norm(u)^2 + norm(g)^2  # Use quadratic for non-zero Hessian
    end

    Qs = [1.0, 2.0]
    times = [1, traj.N]
    params = [nothing, nothing]

    OBJ = GlobalKnotPointObjective((ug, _) -> ℓ(ug), [:u], [:g], traj, params; times=times, Qs=Qs)

    test_objective(OBJ, traj)
end