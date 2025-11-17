export KnotPointObjective
export TerminalObjective

using TrajectoryIndexingUtils

# ----------------------------------------------------------------------------- #
# KnotPointObjective
# ----------------------------------------------------------------------------- #

"""
    KnotPointObjective <: AbstractObjective

Knot point summed objective function for trajectory optimization.

Stores the objective function ℓ that operates on extracted variable values:
```math
J = \\sum_{k \\in \\text{times}} Q_k \\ell(x_k, p_k)
```

where ℓ is evaluated on trajectory variables at each knot point.

# Fields
- `ℓ::Function`: Objective function mapping (variables..., params) -> scalar cost
- `var_names::Vector{Symbol}`: Names of trajectory variables the objective depends on
- `times::Vector{Int}`: Time indices where objective is evaluated
- `params::Vector`: Parameters for each time index
- `Qs::Vector{Float64}`: Weights for each time index
- `∂²Ls::Vector{SparseMatrixCSC{Float64, Int}}`: Preallocated sparse Hessian storage (one per timestep)

# Constructor
```julia
KnotPointObjective(
    ℓ::Function,
    names::Union{Symbol, AbstractVector{Symbol}},
    traj::NamedTrajectory,
    params::AbstractVector;
    times::AbstractVector{Int}=1:traj.N,
    Qs::AbstractVector{Float64}=ones(length(times))
)
```

For single variable: `ℓ(x, p)` where `x` is the variable values at a knot point
For multiple variables: `ℓ(x, u, p)` where each argument corresponds to a variable in `names`

# Examples
```julia
# Single variable
obj = KnotPointObjective((x, _) -> norm(x)^2, :x, traj, fill(nothing, traj.N))

# Multiple variables - concatenated
obj = KnotPointObjective((xu, _) -> xu[1]^2 + xu[2]^2, [:x, :u], traj, fill(nothing, traj.N))

# With parameters and weights
obj = KnotPointObjective(
    (x, p) -> norm(x - p)^2, :x, traj, [x_targets...];
    times=1:10, Qs=[1.0, 2.0, ...]
)
```
"""
struct KnotPointObjective <: AbstractObjective
    ℓ::Function
    var_names::Vector{Symbol}
    times::Vector{Int}
    params::Vector
    Qs::Vector{Float64}
end

function KnotPointObjective(
    ℓ::Function,
    names::AbstractVector{Symbol},
    traj::NamedTrajectory,
    params::AbstractVector;
    times::AbstractVector{Int}=1:traj.N,
    Qs::AbstractVector{Float64}=ones(length(times))
)
    @assert length(Qs) == length(times) "Qs must have the same length as times"
    @assert length(params) == length(times) "params must have the same length as times"

    return KnotPointObjective(
        ℓ,
        Vector{Symbol}(names),
        Vector{Int}(times),
        Vector(params),
        Vector{Float64}(Qs)
    )
end

function KnotPointObjective(
    ℓ::Function,
    names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    times::AbstractVector{Int}=1:traj.N,
    kwargs...
)
    # No params version - create dummy params
    params = [nothing for _ in times]
    ℓ_param = (x, _) -> ℓ(x)
    return KnotPointObjective(ℓ_param, names, traj, params; times=times, kwargs...)
end

function KnotPointObjective(ℓ::Function, name::Symbol, traj::NamedTrajectory; kwargs...)
    return KnotPointObjective(ℓ, [name], traj; kwargs...)
end

function KnotPointObjective(ℓ::Function, name::Symbol, traj::NamedTrajectory, params::AbstractVector; kwargs...)
    return KnotPointObjective(ℓ, [name], traj, params; kwargs...)
end

function TerminalObjective(
    ℓ::Function,
    name::Symbol,
    traj::NamedTrajectory;
    Q::Float64=1.0,
    kwargs...
)
    return KnotPointObjective(
        ℓ,
        name,
        traj;
        Qs=[Q],
        times=[traj.N],
        kwargs...
    )
end

# Implement AbstractObjective interface

function objective_value(obj::KnotPointObjective, traj::NamedTrajectory)
    J = 0.0
    for (i, t) in enumerate(obj.times)
        zₖ = traj[t]
        # Extract relevant variables
        x_vals = vcat([zₖ[name] for name in obj.var_names]...)
        J += obj.Qs[i] * obj.ℓ(x_vals, obj.params[i])
    end
    return J
end

function gradient!(∇::AbstractVector, obj::KnotPointObjective, traj::NamedTrajectory)
    fill!(∇, 0.0)
    
    for (i, k) in enumerate(obj.times)
        zₖ = traj[k]
        # Extract relevant variables and their components
        x_vals = vcat([zₖ[name] for name in obj.var_names]...)
        x_comps = vcat([zₖ.components[name] for name in obj.var_names]...)
        
        # Get indices for this knot point
        knot_indices = slice(k, x_comps, traj.dim)
        
        # Compute gradient directly into view of the gradient vector
        ∇_view = @view ∇[knot_indices]
        ForwardDiff.gradient!(
            ∇_view,
            x -> obj.ℓ(x, obj.params[i]),
            x_vals
        )
        
        # Scale by weight
        ∇_view .*= obj.Qs[i]
    end
    
    return nothing
end

function hessian_structure(obj::KnotPointObjective, traj::NamedTrajectory)

    Z_dim = traj.dim * traj.N + traj.global_dim

    structure = spzeros(Z_dim, Z_dim)

    x_comps = vcat([traj.components[name] for name in obj.var_names]...)

    for k ∈ obj.times
        knot_indices = slice(k, x_comps, traj.dim)
        structure[knot_indices, knot_indices] .= 1.0
    end

    return structure
end

function get_full_hessian(obj::KnotPointObjective, traj::NamedTrajectory)

    Z_dim = traj.dim * traj.N + traj.global_dim

    ∂²L = spzeros(Z_dim, Z_dim)

    x_comps = vcat([traj.components[name] for name in obj.var_names]...)
    
    for (i, k) in enumerate(obj.times)
        zₖ = traj[k]
        knot_indices = slice(k, x_comps, traj.dim)

        ForwardDiff.hessian!(
            view(∂²L, knot_indices, knot_indices),
            x -> obj.Qs[i] * obj.ℓ(x, obj.params[i]),
            vcat([zₖ[name] for name in obj.var_names]...)
        )
    end
    
    return triu(∂²L)
end


# ============================================================================ #

@testitem "testing KnotPointObjective" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    L(a) = norm(a)^2  # Use quadratic for non-zero Hessian
    Qs = [1.0, 2.0]
    times = [1, traj.N]

    OBJ = KnotPointObjective(L, :u, traj, times=times, Qs=Qs)

    test_objective(OBJ, traj; show_hessian_diff=false)
end

@testitem "testing KnotPointObjective with parameters" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    L(x, p) = norm(x)^2 + p  # Use quadratic for non-zero Hessian
    Qs = [1.0, 2.0]
    times = [1, traj.N]
    params = [1.0, 2.0]

    OBJ = KnotPointObjective(L, :u, traj, params; times=times, Qs=Qs)

    test_objective(OBJ, traj)
end