export NonlinearGlobalConstraint

using ..Constraints

# ----------------------------------------------------------------------------- #
# NonlinearGlobalConstraint
# ----------------------------------------------------------------------------- #

"""
    NonlinearGlobalConstraint{F} <: AbstractNonlinearConstraint

Constraint applied to global (trajectory-wide) parameters only.

Stores compact sparse Jacobians/Hessians over global variables only.
Call `jacobian!(constraint, traj)` and `hessian_of_lagrangian!(constraint, traj, μ)` to compute.
Call `get_full_jacobian(constraint, traj)` to assemble full trajectory structure.

# Fields
- `g::F`: Constraint function mapping global variables -> constraint values
- `global_names::Vector{Symbol}`: Names of global variables the constraint depends on
- `equality::Bool`: If true, g(globals) = 0; if false, g(globals) ≤ 0
- `dim::Int`: Dimension of constraint output
- `global_dim::Int`: Combined dimension of all constrained global variables
- `∂g::SparseMatrixCSC{Float64, Int}`: Jacobian storage (dim × global_dim)
- `μ∂²g::SparseMatrixCSC{Float64, Int}`: Hessian storage (global_dim × global_dim)
"""
struct NonlinearGlobalConstraint{F} <: AbstractNonlinearConstraint
    g::F
    global_names::Vector{Symbol}
    equality::Bool
    dim::Int
    global_dim::Int
    ∂g::SparseMatrixCSC{Float64, Int}
    μ∂²g::SparseMatrixCSC{Float64, Int}

    """
        NonlinearGlobalConstraint(
            g::Function,
            global_names::Union{Symbol, AbstractVector{Symbol}},
            traj::NamedTrajectory;
            equality::Bool=true,
            jacobian_structure::Union{Nothing, SparseMatrixCSC{Float64, Int}}=nothing,
            hessian_structure::Union{Nothing, SparseMatrixCSC{Float64, Int}}=nothing
        )

    Create a NonlinearGlobalConstraint object with global components.

    # Arguments
    - `g::Function`: Function over global variable(s) that defines the constraint, `g(globals)`.
    - `global_names::Union{Symbol, AbstractVector{Symbol}}`: Name(s) of the global variable(s) to be constrained.
    - `traj::NamedTrajectory`: The trajectory on which the constraint is defined.

    # Keyword Arguments
    - `equality::Bool=true`: If `true`, the constraint is `g(x) = 0`. Otherwise, the constraint is `g(x) ≤ 0`.
    - `jacobian_structure::Union{Nothing, SparseMatrixCSC{Float64, Int}}=nothing`: Custom sparsity pattern for Jacobian (dim × global_dim)
    - `hessian_structure::Union{Nothing, SparseMatrixCSC{Float64, Int}}=nothing`: Custom sparsity pattern for Hessian (global_dim × global_dim)
    """
    function NonlinearGlobalConstraint(
        g::Function,
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        equality::Bool=true,
        jacobian_structure::Union{Nothing, SparseMatrixCSC{Float64, Int}}=nothing,
        hessian_structure::Union{Nothing, SparseMatrixCSC{Float64, Int}}=nothing
    )
        global_comps = vcat([traj.global_components[name] for name in global_names]...)
        global_dim = length(global_comps)

        # Determine constraint dimension
        g_eval = g(traj.global_data[global_comps])
        @assert g_eval isa AbstractVector{<:Real}
        g_dim = length(g_eval)

        # Initialize Jacobian storage
        if jacobian_structure === nothing
            ∂g = sparse(ones(g_dim, global_dim))
        else
            @assert size(jacobian_structure) == (g_dim, global_dim)
            ∂g = copy(jacobian_structure)
        end

        # Initialize Hessian storage
        if hessian_structure === nothing
            μ∂²g = sparse(ones(global_dim, global_dim))
        else
            @assert size(hessian_structure) == (global_dim, global_dim)
            μ∂²g = copy(hessian_structure)
        end

        return new{typeof(g)}(
            g,
            global_names,
            equality,
            g_dim,
            global_dim,
            ∂g,
            μ∂²g
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

"""
    jacobian!(constraint::NonlinearGlobalConstraint, traj::NamedTrajectory)

Compute Jacobian and store in constraint.∂g (dim × global_dim).
"""
function Constraints.jacobian!(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory
)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    
    # Compute Jacobian directly into sparse storage
    ForwardDiff.jacobian!(
        constraint.∂g,
        x -> constraint.g(x),
        traj.global_data[global_comps]
    )
    return nothing
end

"""
    hessian_of_lagrangian!(constraint::NonlinearGlobalConstraint, traj::NamedTrajectory, μ::AbstractVector)

Compute Hessian weighted by Lagrange multipliers and store in constraint.μ∂²g (global_dim × global_dim).
"""
function Constraints.hessian_of_lagrangian!(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory,
    μ::AbstractVector
)
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    
    # Compute Hessian directly into sparse storage
    ForwardDiff.hessian!(
        constraint.μ∂²g,
        x -> μ' * constraint.g(x),
        traj.global_data[global_comps]
    )
    return nothing
end

# ----------------------------------------------------------------------------- #
# Full Jacobian and Hessian Assembly (mapped to trajectory structure)
# ----------------------------------------------------------------------------- #

"""
    get_full_jacobian(constraint::NonlinearGlobalConstraint, traj::NamedTrajectory)

Assemble full sparse Jacobian (dim × Z_dim) with non-zeros only in global variable columns.
"""
function Constraints.get_full_jacobian(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂g_full = spzeros(constraint.dim, Z_dim)
    
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    offset_global_comps = traj.dim * traj.N .+ global_comps
    
    # Map compact Jacobian to global columns
    ∂g_full[:, offset_global_comps] = constraint.∂g
    
    return ∂g_full
end

"""
    get_full_hessian(constraint::NonlinearGlobalConstraint, traj::NamedTrajectory)

Assemble full Hessian (Z_dim × Z_dim) from compact (global_dim × global_dim) block.
"""
function Constraints.get_full_hessian(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.N + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    
    global_comps = vcat([traj.global_components[name] for name in constraint.global_names]...)
    offset_global_comps = traj.dim * traj.N .+ global_comps
    
    # Map compact Hessian to global×global block
    μ∂²g_full[offset_global_comps, offset_global_comps] = constraint.μ∂²g
    
    return μ∂²g_full
end

function Constraints.jacobian_structure(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory
)
    return get_full_jacobian(constraint, traj)
end

function Constraints.hessian_structure(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory
)
    return get_full_hessian(constraint, traj)
end

# ============================================================================ #

@testitem "testing NonlinearGlobalConstraint" begin    
    include("../../../test/test_utils.jl")

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
    Constraints.jacobian!(NLC, traj)
    ∂g_full = Constraints.get_full_jacobian(NLC, traj)
    ∂g_autodiff = ForwardDiff.jacobian(ĝ, vec(traj))

    @test ∂g_full ≈ ∂g_autodiff

    # Test hessian_of_lagrangian
    μ = randn(g_dim)
    Constraints.hessian_of_lagrangian!(NLC, traj, μ)
    μ∂²g_full = Constraints.get_full_hessian(NLC, traj)
    μ_func = Z -> μ' * ĝ(Z)
    hessian_autodiff = ForwardDiff.hessian(μ_func, vec(traj))

    @test μ∂²g_full ≈ hessian_autodiff
end
