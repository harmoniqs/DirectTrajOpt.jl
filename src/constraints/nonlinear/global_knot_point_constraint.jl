export NonlinearGlobalKnotPointConstraint

using ..Constraints

# --------------------------------------------------------------------------- #
# NonlinearGlobalKnotPointConstraint
# --------------------------------------------------------------------------- #

"""
    NonlinearGlobalKnotPointConstraint{F} <: AbstractNonlinearConstraint

Constraint applied at individual knot points that also depends on global parameters.

Stores constraint function g, variable names, and pre-allocated storage for Jacobians/Hessians.
Each stored Jacobian is (g_dim × combined_var_dim) where combined_var_dim = var_dim + global_dim.

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
- `∂gs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated Jacobian storage (g_dim × combined_dim per timestep)
- `μ∂²gs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated Hessian storage (combined_dim × combined_dim per timestep)
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
    ∂gs::Vector{SparseMatrixCSC{Float64, Int}}
    μ∂²gs::Vector{SparseMatrixCSC{Float64, Int}}

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
        equality::Bool=true,
        jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
        hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
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

        # Pre-allocate storage using provided structures or default sparse matrices filled with ones
        if !isnothing(jacobian_structure)
            @assert size(jacobian_structure) == (g_dim, combined_dim) "jacobian_structure must be (g_dim=$g_dim × combined_dim=$combined_dim)"
            ∂gs = [copy(jacobian_structure) for _ in times]
        else
            # Default: sparse matrix filled with ones
            ∂g_default = sparse(ones(g_dim, combined_dim))
            ∂gs = [copy(∂g_default) for _ in times]
        end

        if !isnothing(hessian_structure)
            @assert size(hessian_structure) == (combined_dim, combined_dim) "hessian_structure must be (combined_dim=$combined_dim × combined_dim=$combined_dim)"
            μ∂²gs = [copy(hessian_structure) for _ in times]
        else
            # Default: sparse matrix filled with ones
            μ∂²g_default = sparse(ones(combined_dim, combined_dim))
            μ∂²gs = [copy(μ∂²g_default) for _ in times]
        end

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
            g_dim * length(times),
            ∂gs,
            μ∂²gs
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

function Constraints.constraint_value(
    constraint::NonlinearGlobalKnotPointConstraint,
    traj::NamedTrajectory
)
    δ = zeros(constraint.dim)
    
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    
    @views for (i, t) ∈ enumerate(constraint.times)
        zₖ = traj[t]
        xg_data = vcat(zₖ.data[x_comps], traj.global_data[global_comps])
        δ[slice(i, constraint.g_dim)] = constraint.g(xg_data, constraint.params[i])
    end
    
    return δ
end

function Constraints.jacobian_structure(
    constraint::NonlinearGlobalKnotPointConstraint,
    traj::NamedTrajectory
)
    # Return structure for single timestep (used by some interfaces)
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    
    # Combined dimension includes knot point vars + global vars
    xg_comps = vcat(x_comps, traj.dim .+ global_comps)
    z_dim = traj.dim + traj.global_dim
    
    ∂g = spzeros(constraint.g_dim, z_dim)
    ∂g[:, xg_comps] .= 1.0
    
    return ∂g
end

function Constraints.hessian_structure(
    constraint::NonlinearGlobalKnotPointConstraint,
    traj::NamedTrajectory
)
    # Return structure for single timestep (used by some interfaces)
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    xg_comps = vcat(x_comps, traj.dim .+ global_comps)
    
    z_dim = traj.dim + traj.global_dim
    μ∂²g = spzeros(z_dim, z_dim)
    μ∂²g[xg_comps, xg_comps] .= 1.0
    
    return μ∂²g
end

"""
    jacobian!(constraint::NonlinearGlobalKnotPointConstraint, traj::NamedTrajectory)

Compute all Jacobians and store them in constraint.∂gs. Each stored Jacobian is (g_dim × combined_dim).
"""
function Constraints.jacobian!(
    constraint::NonlinearGlobalKnotPointConstraint,
    traj::NamedTrajectory
)
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    
    @views for (i, t) ∈ enumerate(constraint.times)
        zₖ = traj[t]
        # Combine knot point and global data
        xg_data = vcat(zₖ.data[x_comps], traj.global_data[global_comps])
        
        # Compute Jacobian directly into sparse storage
        ForwardDiff.jacobian!(
            constraint.∂gs[i],
            x -> constraint.g(x, constraint.params[i]),
            xg_data
        )
    end
    return nothing
end

"""
    hessian_of_lagrangian!(constraint::NonlinearGlobalKnotPointConstraint, traj::NamedTrajectory, μ::AbstractVector)

Compute Hessian weighted by Lagrange multipliers for constraint with global and knot point variables.
Each stored Hessian is (combined_dim × combined_dim).
"""
function Constraints.hessian_of_lagrangian!(
    constraint::NonlinearGlobalKnotPointConstraint,
    traj::NamedTrajectory,
    μ::AbstractVector
)
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    
    @views for (i, t) ∈ enumerate(constraint.times)
        zₖ = traj[t]
        μₖ = μ[slice(i, constraint.g_dim)]
        
        # Combine knot point and global data
        xg_data = vcat(zₖ.data[x_comps], traj.global_data[global_comps])
        
        # Compute Hessian directly into sparse storage
        ForwardDiff.hessian!(
            constraint.μ∂²gs[i],
            x -> μₖ' * constraint.g(x, constraint.params[i]),
            xg_data
        )
    end
    return nothing
end

# ----------------------------------------------------------------------------- #
# Full Jacobian and Hessian Assembly
# ----------------------------------------------------------------------------- #

"""
    get_full_jacobian(constraint::NonlinearGlobalKnotPointConstraint, traj::NamedTrajectory)

Assemble full sparse Jacobian from stored (g_dim × combined_dim) blocks.
Each block corresponds to one timestep and has non-zero entries for both knot point and global variables.
"""
function Constraints.get_full_jacobian(
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
        
        # Map stored Jacobian to full structure
        # First var_dim columns map to knot point variables at time t
        col_range_knot = slice(t, x_comps, traj.dim)
        ∂g_full[row_range, col_range_knot] = constraint.∂gs[i][:, 1:constraint.var_dim]
        
        # Remaining columns map to global variables
        ∂g_full[row_range, offset_global_comps] = constraint.∂gs[i][:, constraint.var_dim+1:end]
    end
    
    return ∂g_full
end

"""
    get_full_hessian(constraint::NonlinearGlobalKnotPointConstraint, traj::NamedTrajectory)

Assemble full sparse Hessian from stored (combined_dim × combined_dim) blocks.
Accumulates contributions since global variables appear in multiple timesteps.
"""
function Constraints.get_full_hessian(
    constraint::NonlinearGlobalKnotPointConstraint,
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.N + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    
    x_comps = vcat([traj.components[name] for name in constraint.var_names]...)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    offset_global_comps = traj.dim * traj.N .+ global_comps
    
    @views for (i, t) ∈ enumerate(constraint.times)
        # Map stored Hessian to full structure
        knot_range = slice(t, x_comps, traj.dim)
        
        # Knot × Knot block
        μ∂²g_full[knot_range, knot_range] .+= constraint.μ∂²gs[i][1:constraint.var_dim, 1:constraint.var_dim]
        
        # Knot × Global block
        μ∂²g_full[knot_range, offset_global_comps] .+= constraint.μ∂²gs[i][1:constraint.var_dim, constraint.var_dim+1:end]
        
        # Global × Knot block
        μ∂²g_full[offset_global_comps, knot_range] .+= constraint.μ∂²gs[i][constraint.var_dim+1:end, 1:constraint.var_dim]
        
        # Global × Global block (accumulated across timesteps)
        μ∂²g_full[offset_global_comps, offset_global_comps] .+= constraint.μ∂²gs[i][constraint.var_dim+1:end, constraint.var_dim+1:end]
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

    g_dim = 2
    times = 1:traj.N

    NLC = NonlinearGlobalKnotPointConstraint(g_fn, [:u], [:g], traj; times=times, equality=false)
    U_DIM(k) = slice(k, traj.components[:u], traj.dim)
    G_DIM = traj.dim * traj.N .+ traj.global_components[:g]

    ĝ(Z⃗) = vcat([g_fn(Z⃗[vcat(U_DIM(k), G_DIM)]) for k ∈ times]...)

    # Test constraint_value
    δ = Constraints.constraint_value(NLC, traj)
    @test δ ≈ ĝ(vec(traj))

    # Test jacobian!
    Constraints.jacobian!(NLC, traj)
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full ≈ ∂g_autodiff

    # Test hessian_of_lagrangian
    μ = randn(g_dim * traj.N)
    Constraints.hessian_of_lagrangian!(NLC, traj, μ)
    μ∂²g_full = Constraints.get_full_hessian(NLC, traj)
    
    μ_func = Z -> μ' * ĝ(Z)
    hessian_autodiff = ForwardDiff.hessian(μ_func, vec(traj))

    @test μ∂²g_full ≈ hessian_autodiff
end
