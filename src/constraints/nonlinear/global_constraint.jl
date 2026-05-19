export NonlinearGlobalConstraint

using ..Constraints

# ----------------------------------------------------------------------------- #
# NonlinearGlobalConstraint
# ----------------------------------------------------------------------------- #

"""
    NonlinearGlobalConstraint{F} <: AbstractNonlinearConstraint

Constraint applied to global (trajectory-wide) parameters only.

Computes Jacobians and Hessians on-the-fly using automatic differentiation.
For pre-allocated optimization, see Piccolissimo.OptimizedNonlinearGlobalConstraint.

# Fields
- `g::F`: Constraint function mapping global variables -> constraint values
- `global_names::Vector{Symbol}`: Names of global variables the constraint depends on
- `equality::Bool`: If true, g(globals) = 0; if false, g(globals) ≤ 0
- `dim::Int`: Dimension of constraint output
- `global_dim::Int`: Combined dimension of all constrained global variables
"""
struct NonlinearGlobalConstraint{F} <: AbstractNonlinearConstraint
    g::F
    global_names::Vector{Symbol}
    equality::Bool
    dim::Int
    global_dim::Int

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
    """
    function NonlinearGlobalConstraint(
        g::Function,
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        equality::Bool = true,
    )
        global_comps = vcat([traj.global_components[name] for name in global_names]...)
        global_dim = length(global_comps)

        # Determine constraint dimension
        g_eval = g(traj.global_data[global_comps])
        @assert g_eval isa AbstractVector{<:Real}
        g_dim = length(g_eval)

        return new{typeof(g)}(g, global_names, equality, g_dim, global_dim)
    end
end

function NonlinearGlobalConstraint(
    g::Function,
    global_name::Symbol,
    traj::NamedTrajectory;
    kwargs...,
)
    return NonlinearGlobalConstraint(g, [global_name], traj; kwargs...)
end

function Base.show(io::IO, c::NonlinearGlobalConstraint)
    globals = join([":$n" for n in c.global_names], ", ")
    eq_str = c.equality ? "equality" : "inequality"
    print(io, "NonlinearGlobalConstraint on [$globals], $eq_str (dim = $(c.dim))")
end

# ----------------------------------------------------------------------------- #
# Method Implementations for NonlinearGlobalConstraint
# ----------------------------------------------------------------------------- #

# Per-constraint evaluation methods (called by wrapper functions)
function (constraint::NonlinearGlobalConstraint)(δ::AbstractVector, traj::NamedTrajectory)
    global_comps =
        vcat([traj.global_components[name] for name in constraint.global_names]...)
    δ .= constraint.g(traj.global_data[global_comps])
    return nothing
end

"""
    evaluate!(values::AbstractVector, constraint::NonlinearGlobalConstraint, traj::NamedTrajectory)

Evaluate the global constraint, storing results in-place in `values`.
This is part of the common interface with integrators.
"""
function CommonInterface.evaluate!(
    values::AbstractVector,
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory,
)
    global_comps =
        vcat([traj.global_components[name] for name in constraint.global_names]...)
    values .= constraint.g(traj.global_data[global_comps])
    return nothing
end

"""
    eval_jacobian(constraint::NonlinearGlobalConstraint, traj::NamedTrajectory)

Compute and return the full Jacobian using automatic differentiation.
"""
function CommonInterface.eval_jacobian(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory,
)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂g_full = spzeros(constraint.dim, Z_dim)

    global_comps =
        vcat([traj.global_components[name] for name in constraint.global_names]...)
    offset_global_comps = traj.dim * traj.N .+ global_comps

    # Compute compact Jacobian and map to global columns
    ∂g_compact = ForwardDiff.jacobian(x -> constraint.g(x), traj.global_data[global_comps])
    ∂g_full[:, offset_global_comps] = ∂g_compact

    return ∂g_full
end

"""
    eval_hessian_of_lagrangian(constraint::NonlinearGlobalConstraint, traj::NamedTrajectory, μ::AbstractVector)

Compute and return the full Hessian of the Lagrangian using automatic differentiation.
"""
function CommonInterface.eval_hessian_of_lagrangian(
    constraint::NonlinearGlobalConstraint,
    traj::NamedTrajectory,
    μ::AbstractVector,
)
    Z_dim = traj.dim * traj.N + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)

    global_comps =
        vcat([traj.global_components[name] for name in constraint.global_names]...)
    offset_global_comps = traj.dim * traj.N .+ global_comps

    # Compute compact Hessian and map to global×global block
    μ∂²g_compact =
        ForwardDiff.hessian(x -> μ' * constraint.g(x), traj.global_data[global_comps])
    μ∂²g_full[offset_global_comps, offset_global_comps] = μ∂²g_compact

    return μ∂²g_full
end

# ============================================================================ #

@testitem "testing NonlinearGlobalConstraint" begin
    include("../../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory(add_global = true)

    g_fn(g) = [norm(g) - 1.0]

    NLC = NonlinearGlobalConstraint(g_fn, :g, traj; equality = false)

    # Test with validation utility
    test_constraint(
        NLC,
        traj;
        atol = 1e-3,
        show_jacobian_diff = false,
        show_hessian_diff = false,
    )
end
