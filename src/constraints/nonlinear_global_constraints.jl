export NonlinearGlobalConstraint
export NonlinearGlobalKnotPointConstraint

using ..Constraints

# ----------------------------------------------------------------------------- #
# NonlinearGlobalConstraint
# ----------------------------------------------------------------------------- #

"""
    NonlinearGlobalConstraint{F, ∂G, μ∂²G} <: AbstractNonlinearConstraint

Constraint applied to global (trajectory-wide) parameters only.

# Fields
- `g::F`: Constraint function mapping global variables -> constraint values
- `global_names::Vector{Symbol}`: Names of global variables the constraint depends on
- `equality::Bool`: If true, g(globals) = 0; if false, g(globals) ≤ 0
- `dim::Int`: Dimension of constraint output
- `∂g!::∂G`: Wrapper function for Jacobian evaluation
- `∂gs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated Jacobian storage (vector for consistency with evaluator)
- `μ∂²g!::μ∂²G`: Wrapper function for Hessian evaluation
- `μ∂²gs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated Hessian storage (vector for consistency with evaluator)
"""
struct NonlinearGlobalConstraint{F, ∂G, μ∂²G} <: AbstractNonlinearConstraint
    g::F
    global_names::Vector{Symbol}
    equality::Bool
    dim::Int
    ∂g!::∂G
    ∂gs::Vector{SparseMatrixCSC{Float64, Int}}
    μ∂²g!::μ∂²G
    μ∂²gs::Vector{SparseMatrixCSC{Float64, Int}}

    """
        NonlinearGlobalConstraint(
            g::Function,
            global_names::Union{Symbol, AbstractVector{Symbol}},
            traj::NamedTrajectory;
            equality::Bool=true
        )

    Create a NonlinearGlobalConstraint object with global components.

    # Arguments
    - `g::Function`: Function over global variable(s) that defines the constraint, `g(globals)`.
    - `global_names::Union{Symbol, AbstractVector{Symbol}}`: Name(s) of the global variable(s) to be constrained.
    - `traj::NamedTrajectory`: The trajectory on which the constraint is defined.

    # Keyword Arguments
    - `equality::Bool=true`: If `true`, the constraint is `g(x) = 0`. Otherwise, the constraint is `g(x) ≤ 0`.
    """
    function NonlinearGlobalConstraint(
        g::Function,
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        equality::Bool=true,
    )
        global_comps = vcat([traj.global_components[name] for name in global_names]...)

        # Determine constraint dimension
        g_eval = g(traj.global_data[global_comps])
        @assert g_eval isa AbstractVector{<:Real}
        g_dim = length(g_eval)

        # Create sparsity structures
        Z_dim = traj.dim * traj.N + traj.global_dim
        
        # Jacobian structure (g_dim × Z_dim, only global columns are non-zero)
        ∂g_structure = spzeros(g_dim, Z_dim)
        offset_global_comps = traj.dim * traj.N .+ global_comps
        ∂g_structure[:, offset_global_comps] .= 1.0
        ∂gs = [copy(∂g_structure)]  # Vector for consistency with evaluator

        # Hessian structure (Z_dim × Z_dim, only global×global block is non-zero)
        μ∂²g_structure = spzeros(Z_dim, Z_dim)
        μ∂²g_structure[offset_global_comps, offset_global_comps] .= 1.0
        μ∂²gs = [copy(μ∂²g_structure)]  # Vector for consistency with evaluator

        # Create wrapper functions
        function ∂g!(
            ∂gs::Vector{SparseMatrixCSC{Float64, Int}},
            Z⃗::AbstractVector
        )
            # Extract global data from Z⃗
            # Z⃗ = [trajectory_data..., global_data...]
            global_data = Z⃗[traj.dim * traj.N .+ (1:traj.global_dim)]
            
            # Compute Jacobian in global subspace
            ∂g_local = zeros(g_dim, length(global_comps))
            ForwardDiff.jacobian!(
                ∂g_local,
                x -> g(x),
                global_data[global_comps]
            )
            
            # Map to full structure
            fill!(∂gs[1], 0.0)
            ∂gs[1][:, offset_global_comps] = ∂g_local
            
            return nothing
        end

        function μ∂²g!(
            μ∂²gs::Vector{SparseMatrixCSC{Float64, Int}},
            Z⃗::AbstractVector,
            μ⃗::AbstractVector
        )
            # Extract global data from Z⃗
            # Z⃗ = [trajectory_data..., global_data...]
            global_data = Z⃗[traj.dim * traj.N .+ (1:traj.global_dim)]
            
            # Compute Hessian in global subspace
            μ∂²g_local = zeros(length(global_comps), length(global_comps))
            ForwardDiff.hessian!(
                μ∂²g_local,
                x -> μ⃗' * g(x),
                global_data[global_comps]
            )
            
            # Map to full structure
            fill!(μ∂²gs[1], 0.0)
            μ∂²gs[1][offset_global_comps, offset_global_comps] = μ∂²g_local
            
            return nothing
        end

        return new{typeof(g), typeof(∂g!), typeof(μ∂²g!)}(
            g,
            global_names,
            equality,
            g_dim,
            ∂g!,
            ∂gs,
            μ∂²g!,
            μ∂²gs
        )
    end
end

function NonlinearGlobalConstraint(
    g::Function,
    global_name::Symbol,
    traj::NamedTrajectory;
    kwargs...
)
    return NonlinearGlobalConstraint(g, [global_name], traj; kwargs...)
end

# ----------------------------------------------------------------------------- #
# Method Implementations for NonlinearGlobalConstraint
# ----------------------------------------------------------------------------- #

# Per-constraint evaluation methods (called by wrapper functions)
function (constraint::NonlinearGlobalConstraint)(
    δ::AbstractVector,
    traj::NamedTrajectory
)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    δ .= constraint.g(traj.global_data[global_comps])
    return nothing
end

function Constraints.constraint_value(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory
)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    return constraint.g(traj.global_data[global_comps])
end

function Constraints.jacobian_structure(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory
)
    return constraint.∂gs[1]
end

function Constraints.hessian_structure(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory
)
    return constraint.μ∂²gs[1]
end

function get_full_jacobian(
    NLC::NonlinearGlobalConstraint, 
    traj::NamedTrajectory
)
    NLC.∂g!(NLC.∂gs, vec(traj))
    return NLC.∂gs[1]
end

function get_full_hessian(
    NLC::NonlinearGlobalConstraint, 
    traj::NamedTrajectory
)
    μ = ones(NLC.dim)  # Dummy multipliers for structure
    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)
    return NLC.μ∂²gs[1]
end

# --------------------------------------------------------------------------- #
# NonlinearGlobalKnotPointConstraint
# --------------------------------------------------------------------------- #

"""
    NonlinearGlobalKnotPointConstraint{F, ∂G, μ∂²G} <: AbstractNonlinearConstraint

Constraint applied at individual knot points that also depends on global parameters.

# Fields
- `g::F`: Constraint function mapping (knot_point_vars..., global_vars..., params) -> constraint values
- `var_names::Vector{Symbol}`: Names of knot point variables the constraint depends on
- `global_names::Vector{Symbol}`: Names of global variables the constraint depends on
- `times::Vector{Int}`: Time indices where constraint is applied
- `equality::Bool`: If true, g(x, globals) = 0; if false, g(x, globals) ≤ 0
- `params::Vector`: Parameters for each time index
- `g_dim::Int`: Dimension of constraint output at each time step
- `dim::Int`: Total constraint dimension (g_dim * length(times))
- `∂g!::∂G`: Wrapper function for Jacobian evaluation
- `∂gs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated Jacobian storage (one per time step)
- `μ∂²g!::μ∂²G`: Wrapper function for Hessian evaluation
- `μ∂²gs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated Hessian storage (one per time step)
"""
struct NonlinearGlobalKnotPointConstraint{F, ∂G, μ∂²G} <: AbstractNonlinearConstraint
    g::F
    var_names::Vector{Symbol}
    global_names::Vector{Symbol}
    times::Vector{Int}
    equality::Bool
    params::Vector
    g_dim::Int
    dim::Int
    ∂g!::∂G
    ∂gs::Vector{SparseMatrixCSC{Float64, Int}}
    μ∂²g!::μ∂²G
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
    )
        @assert length(params) == length(times) "params must have the same length as times"

        # Get component indices
        x_comps = vcat([traj.components[name] for name in names]...)
        global_comps = vcat([traj.global_components[name] for name in global_names]...)
        offset_global_comps = traj.dim * traj.N .+ global_comps

        # Determine constraint dimension by evaluating with first parameter
        Z⃗ = vec(traj)
        xg_slice_test = vcat(slice(1, x_comps, traj.dim), offset_global_comps)
        @assert g(Z⃗[xg_slice_test], params[1]) isa AbstractVector{<:Real}
        g_dim = length(g(Z⃗[xg_slice_test], params[1]))

        # Create sparsity structures - one per timestep
        Z_dim = traj.dim * traj.N + traj.global_dim
        
        # Jacobian structure (g_dim × Z_dim per timestep)
        # Non-zero at knot point vars and global vars
        ∂g_structure = spzeros(g_dim, Z_dim)
        ∂gs = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(times))
        for (i, t) in enumerate(times)
            ∂g_k = copy(∂g_structure)
            # Mark knot point vars
            xg_slice = vcat(slice(t, x_comps, traj.dim), offset_global_comps)
            ∂g_k[:, xg_slice] .= 1.0
            ∂gs[i] = ∂g_k
        end

        # Hessian structure (Z_dim × Z_dim per timestep)
        # Non-zero at (knot_vars × knot_vars), (knot_vars × global), (global × global)
        μ∂²g_structure = spzeros(Z_dim, Z_dim)
        μ∂²gs = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(times))
        for (i, t) in enumerate(times)
            μ∂²g_k = copy(μ∂²g_structure)
            xg_slice = vcat(slice(t, x_comps, traj.dim), offset_global_comps)
            μ∂²g_k[xg_slice, xg_slice] .= 1.0
            μ∂²gs[i] = μ∂²g_k
        end

        # Create wrapper functions
        function ∂g!(
            ∂gs::Vector{SparseMatrixCSC{Float64, Int}},
            Z⃗::AbstractVector
        )
            # Extract trajectory and global data from Z⃗
            # Z⃗ = [trajectory_data..., global_data...]
            traj_data = Z⃗[1:traj.dim * traj.N]
            global_data = Z⃗[traj.dim * traj.N .+ (1:traj.global_dim)]
            
            # Wrap trajectory data
            Z_traj = NamedTrajectory(traj; datavec=traj_data)
            
            @views for (i, t) ∈ enumerate(times)
                zₖ = Z_traj[t]
                xg_data = vcat(zₖ.data[x_comps], global_data[global_comps])
                
                # Compute Jacobian for this timestep
                ∂g_local = zeros(g_dim, length(x_comps) + length(global_comps))
                ForwardDiff.jacobian!(
                    ∂g_local,
                    x -> g(x, params[i]),
                    xg_data
                )
                
                # Map to full structure
                fill!(∂gs[i], 0.0)
                xg_slice = vcat(slice(t, x_comps, traj.dim), offset_global_comps)
                ∂gs[i][:, xg_slice] = ∂g_local
            end
            
            return nothing
        end

        function μ∂²g!(
            μ∂²gs::Vector{SparseMatrixCSC{Float64, Int}},
            Z⃗::AbstractVector,
            μ⃗::AbstractVector
        )
            # Extract trajectory and global data from Z⃗
            # Z⃗ = [trajectory_data..., global_data...]
            traj_data = Z⃗[1:traj.dim * traj.N]
            global_data = Z⃗[traj.dim * traj.N .+ (1:traj.global_dim)]
            
            # Wrap trajectory data
            Z_traj = NamedTrajectory(traj; datavec=traj_data)
            
            # Initialize all Hessians to zero (they will be accumulated)
            for μ∂²g_k in μ∂²gs
                fill!(μ∂²g_k, 0.0)
            end
            
            @views for (i, t) ∈ enumerate(times)
                zₖ = Z_traj[t]
                xg_data = vcat(zₖ.data[x_comps], global_data[global_comps])
                μ_slice = slice(i, g_dim)
                
                # Compute Hessian for this timestep
                μ∂²g_local = zeros(length(x_comps) + length(global_comps), length(x_comps) + length(global_comps))
                ForwardDiff.hessian!(
                    μ∂²g_local,
                    xg -> μ⃗[μ_slice]' * g(xg, params[i]),
                    xg_data
                )
                
                # Map to full structure (accumulate in global subspace)
                xg_slice = vcat(slice(t, x_comps, traj.dim), offset_global_comps)
                μ∂²gs[i][xg_slice, xg_slice] .= μ∂²g_local
            end
            
            return nothing
        end

        return new{typeof(g), typeof(∂g!), typeof(μ∂²g!)}(
            g,
            names,
            global_names,
            times,
            equality,
            params,
            g_dim,
            g_dim * length(times),
            ∂g!,
            ∂gs,
            μ∂²g!,
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

function get_full_jacobian(
    NLC::NonlinearGlobalKnotPointConstraint, 
    traj::NamedTrajectory
)
    # Use wrapper function with pre-allocated storage
    NLC.∂g!(NLC.∂gs, vec(traj))
    
    # Aggregate all timestep Jacobians into single matrix
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂g_full = spzeros(NLC.dim, Z_dim)
    
    for (i, t) ∈ enumerate(NLC.times)
        ∂g_full[slice(i, NLC.g_dim), :] = NLC.∂gs[i]
    end
    
    return ∂g_full
end

function get_full_hessian(
    NLC::NonlinearGlobalKnotPointConstraint, 
    traj::NamedTrajectory
)
    μ = ones(NLC.dim)  # Dummy multipliers for structure
    
    # Use wrapper function with pre-allocated storage
    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)
    
    # Aggregate all timestep Hessians (accumulate for overlapping global subspace)
    Z_dim = traj.dim * traj.N + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    
    for μ∂²g_k in NLC.μ∂²gs
        μ∂²g_full .+= μ∂²g_k
    end
    
    return μ∂²g_full
end

# ============================================================================ #

@testitem "testing NonlinearGlobalConstraint" begin    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory(add_global=true)

    g_fn(g) = [norm(g) - 1.0]

    g_dim = 1

    NLC = NonlinearGlobalConstraint(g_fn, :g, traj; equality=false)
    G_DIM = traj.dim * traj.N .+ traj.global_components[:g]

    ĝ(Z⃗) = g_fn(Z⃗[G_DIM])

    # Test constraint_value
    δ = Constraints.constraint_value(NLC, traj)
    @test δ ≈ ĝ(vec(traj))

    # Test jacobian!
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full ≈ ∂g_autodiff

    # Test hessian_of_lagrangian
    μ = randn(g_dim)
    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)
    μ∂²g_full = NLC.μ∂²gs[1]
    μ_func = Z -> μ' * ĝ(Z)
    hessian_autodiff = ForwardDiff.hessian(μ_func, vec(traj))

    @test μ∂²g_full ≈ hessian_autodiff
end

@testitem "testing NonlinearGlobalKnotPointConstraint" begin
    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

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
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full ≈ ∂g_autodiff

    # Test hessian_of_lagrangian
    μ = randn(g_dim * traj.N)
    NLC.μ∂²g!(NLC.μ∂²gs, vec(traj), μ)
    
    # Aggregate all timestep Hessians (accumulate for overlapping global subspace)
    Z_dim = traj.dim * traj.N + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    for μ∂²g_k in NLC.μ∂²gs
        μ∂²g_full .+= μ∂²g_k
    end
    
    μ_func = Z -> μ' * ĝ(Z)
    hessian_autodiff = ForwardDiff.hessian(μ_func, vec(traj))

    @test μ∂²g_full ≈ hessian_autodiff
end