export NonlinearGlobalKnotPointConstraint

using ..Constraints

# --------------------------------------------------------------------------- #
# NonlinearGlobalKnotPointConstraint
# --------------------------------------------------------------------------- #

"""
    NonlinearGlobalKnotPointConstraint{F} <: AbstractNonlinearConstraint

Constraint applied at individual knot points that also depends on global parameters.

Computes Jacobians and Hessians on-the-fly using automatic differentiation.
For pre-allocated optimization, see Piccolissimo.OptimizedNonlinearGlobalKnotPointConstraint.

# Fields
- `g::F`: Constraint function mapping (knot_point_vars..., global_vars..., params) -> constraint values
- `var_names::Vector{Symbol}`: Names of knot point variables the constraint depends on
- `global_names::Vector{Symbol}`: Names of global variables the constraint depends on
- `times::Vector{Int}`: Time indices where constraint is applied
- `equality::Bool`: If true, g(x, globals) = 0; if false, g(x, globals) ≤ 0
- `params::Vector`: Parameters for each time index
- `g_dim::Int`: Dimension of constraint output at each time step
- `var_dim::Int`: Combined dimension of knot point variables
- `global_dim::Int`: Combined dimension of global variables
- `combined_dim::Int`: var_dim + global_dim
- `dim::Int`: Total constraint dimension (g_dim * length(times))
"""
struct NonlinearGlobalKnotPointConstraint{F} <: AbstractNonlinearConstraint
    g::F
    var_names::Vector{Symbol}
    global_names::Vector{Symbol}
    times::Vector{Int}
    equality::Bool
    params::Vector
    g_dim::Int
    var_dim::Int
    global_dim::Int
    combined_dim::Int
    dim::Int

    """
        NonlinearGlobalKnotPointConstraint(
            g::Function,
            names::AbstractVector{Symbol},
            global_names::AbstractVector{Symbol},
            traj::NamedTrajectory,
            params::AbstractVector;
            times::AbstractVector{Int}=1:traj.N,
            equality::Bool=true
        )

    Create a NonlinearKnotPointConstraint object with global components.
    """
    function NonlinearGlobalKnotPointConstraint(
        g::Function,
        names::AbstractVector{Symbol},
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory,
        params::AbstractVector;
        times::AbstractVector{Int}=1:traj.N,
        equality::Bool=true
    )
        @assert length(params) == length(times) "params must have the same length as times"

        # Get component indices
        x_comps = vcat([traj.components[name] for name in names]...)
        global_comps = vcat([traj.global_components[name] for name in global_names]...)
        
        var_dim = length(x_comps)
        global_dim_local = length(global_comps)
        combined_dim = var_dim + global_dim_local

        # Determine constraint dimension by evaluating with first parameter
        # Create test vector combining knot point and global data
        Z⃗ = vec(traj)
        x_slice_test = slice(1, x_comps, traj.dim)
        offset_global_comps = traj.dim * traj.N .+ global_comps
        xg_test = vcat(Z⃗[x_slice_test], Z⃗[offset_global_comps])
        
        @assert g(xg_test, params[1]) isa AbstractVector{<:Real}
        g_dim = length(g(xg_test, params[1]))

        return new{typeof(g)}(
            g,
            names,
            global_names,
            times,
            equality,
            params,
            g_dim,
            var_dim,
            global_dim_local,
            combined_dim,
            g_dim * length(times)
        )
    end
end

function NonlinearGlobalKnotPointConstraint(
    g::Function,
    names::AbstractVector{Symbol},
    global_names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    times::AbstractVector{Int}=1:traj.N,
    kwargs...
)
    params = [nothing for _ in times]
    g_param = (x, _) -> g(x)
    return NonlinearGlobalKnotPointConstraint(
        g_param, 
        names,
        global_names,
        traj, 
        params; 
        times=times, 
        kwargs...
    )
end

function NonlinearGlobalKnotPointConstraint(g::Function, name::Symbol, args...; kwargs...)
    return NonlinearGlobalKnotPointConstraint(g, [name], args...; kwargs...)
end

# ----------------------------------------------------------------------------- #
# Method Implementations for NonlinearGlobalKnotPointConstraint
# ----------------------------------------------------------------------------- #

"""
    evaluate!(values::AbstractVector, constraint::NonlinearGlobalKnotPointConstraint, traj::NamedTrajectory)

Evaluate the global knot point constraint, storing results in-place in `values`.
This is part of the common interface with integrators.
"""
function CommonInterface.evaluate!(
    values::AbstractVector,
    constraint::NonlinearGlobalKnotPointConstraint,
    traj::NamedTrajectory
)
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    
    @views for (i, t) ∈ enumerate(constraint.times)
        zₖ = traj[t]
        xg_data = vcat(zₖ.data[x_comps], traj.global_data[global_comps])
        values[slice(i, constraint.g_dim)] = constraint.g(xg_data, constraint.params[i])
    end
    
    return nothing
end

"""
    eval_jacobian(constraint::NonlinearGlobalKnotPointConstraint, traj::NamedTrajectory)

Compute and return the full Jacobian using automatic differentiation.
"""
function CommonInterface.eval_jacobian(
    constraint::NonlinearGlobalKnotPointConstraint,
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂g_full = spzeros(constraint.dim, Z_dim)
    
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    offset_global_comps = traj.dim * traj.N .+ global_comps
    
    @views for (i, t) ∈ enumerate(constraint.times)
        # Rows: constraint equations for this timestep
        row_range = slice(i, constraint.g_dim)
        
        # Combine knot point and global data
        zₖ = traj[t]
        xg_data = vcat(zₖ.data[x_comps], traj.global_data[global_comps])
        
        # Compute compact Jacobian
        ∂g_compact = ForwardDiff.jacobian(
            x -> constraint.g(x, constraint.params[i]),
            xg_data
        )
        
        # Map to full structure
        # First var_dim columns map to knot point variables at time t
        col_range_knot = slice(t, x_comps, traj.dim)
        ∂g_full[row_range, col_range_knot] = ∂g_compact[:, 1:constraint.var_dim]
        
        # Remaining columns map to global variables
        ∂g_full[row_range, offset_global_comps] = ∂g_compact[:, constraint.var_dim+1:end]
    end
    
    return ∂g_full
end

"""
    eval_hessian_of_lagrangian(constraint::NonlinearGlobalKnotPointConstraint, traj::NamedTrajectory, μ::AbstractVector)

Compute and return the full Hessian of the Lagrangian using automatic differentiation.
"""
function CommonInterface.eval_hessian_of_lagrangian(
    constraint::NonlinearGlobalKnotPointConstraint,
    traj::NamedTrajectory,
    μ::AbstractVector
)
    Z_dim = traj.dim * traj.N + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    offset_global_comps = traj.dim * traj.N .+ global_comps
    
    @views for (i, t) ∈ enumerate(constraint.times)
        # Combine knot point and global data
        zₖ = traj[t]
        xg_data = vcat(zₖ.data[x_comps], traj.global_data[global_comps])
        μₖ = μ[slice(i, constraint.g_dim)]
        
        # Compute compact Hessian
        μ∂²g_compact = ForwardDiff.hessian(
            x -> μₖ' * constraint.g(x, constraint.params[i]),
            xg_data
        )
        
        # Map to full structure with accumulation
        knot_range = slice(t, x_comps, traj.dim)
        
        # Knot × Knot block
        μ∂²g_full[knot_range, knot_range] .+= μ∂²g_compact[1:constraint.var_dim, 1:constraint.var_dim]
        
        # Knot × Global block
        μ∂²g_full[knot_range, offset_global_comps] .+= μ∂²g_compact[1:constraint.var_dim, constraint.var_dim+1:end]
        
        # Global × Knot block
        μ∂²g_full[offset_global_comps, knot_range] .+= μ∂²g_compact[constraint.var_dim+1:end, 1:constraint.var_dim]
        
        # Global × Global block (accumulated across timesteps)
        μ∂²g_full[offset_global_comps, offset_global_comps] .+= μ∂²g_compact[constraint.var_dim+1:end, constraint.var_dim+1:end]
    end
    
    return μ∂²g_full
end

# ============================================================================ #

@testitem "testing NonlinearGlobalKnotPointConstraint" begin
    using TrajectoryIndexingUtils
    
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory(add_global=true)

    function g_fn(ug)
        u, g = ug[1:traj.dims[:u]], ug[traj.dims[:u] + 1:end]
        return [norm(u) - 1.0; norm(u) * norm(g) - 1.0]
    end

    times = 1:traj.N

    NLC = NonlinearGlobalKnotPointConstraint(g_fn, [:u], [:g], traj; times=times, equality=false)

    # Test with validation utility
    test_constraint(NLC, traj; atol=1.5e-2, show_hessian_diff=true)
end
