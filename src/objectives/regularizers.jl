export QuadraticRegularizer

using TrajectoryIndexingUtils

# ----------------------------------------------------------------------------- #
# Quadratic Regularizer
# ----------------------------------------------------------------------------- #

"""
    QuadraticRegularizer <: AbstractObjective

Quadratic regularization objective for a trajectory component.

Computes:
```math
J = \\sum_{k \\in \\text{times}} \\frac{1}{2} (v_k - v_\\text{baseline})^T R (v_k - v_\\text{baseline}) \\Delta t
```

Gradients and Hessians are computed analytically.

# Fields
- `name::Symbol`: Name of the variable to regularize
- `R::Vector{Float64}`: Diagonal weight matrix
- `baseline::Matrix{Float64}`: Baseline values (column per timestep)
- `times::Vector{Int}`: Time indices where regularization is applied
- `∂²L::Matrix{Float64}`: Preallocated Hessian storage

# Constructor
```julia
QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::Union{Real, AbstractVector{<:Real}};
    baseline::AbstractMatrix{<:Real}=zeros(traj.dims[name], traj.N),
    times::AbstractVector{Int}=1:traj.N
)
```
"""
struct QuadraticRegularizer <: AbstractObjective
    name::Symbol
    R::Vector{Float64}
    baseline::Matrix{Float64}
    times::Vector{Int}
    ∂²J::SparseMatrixCSC{Float64, Int}
end

function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::AbstractVector{<:Real};
    baseline::AbstractMatrix{<:Real}=zeros(traj.dims[name], traj.N),
    times::AbstractVector{Int}=1:traj.N,
)
    @assert length(R) == traj.dims[name]
    
    ∂²J = quadratic_regularizer_hessian_structure(traj, name, collect(times))
   
    return QuadraticRegularizer(
        name,
        Vector{Float64}(R),
        Matrix{Float64}(baseline),
        Vector{Int}(times),
        ∂²J
    )
end

function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::Real;
    kwargs...
)
    return QuadraticRegularizer(name, traj, R * ones(traj.dims[name]); kwargs...)
end

# Implement AbstractObjective interface

function objective_value(reg::QuadraticRegularizer, traj::NamedTrajectory)
    J = 0.0
    for t ∈ reg.times
        zₖ = traj[t]
        vₖ = zₖ[reg.name]
        Δv = vₖ - reg.baseline[:, t]
        Δt = zₖ.timestep
        rₖ = Δt .* Δv
        J += 0.5 * rₖ' * (reg.R .* rₖ)
    end
    return J
end

function gradient!(∇::AbstractVector, reg::QuadraticRegularizer, traj::NamedTrajectory)
    v_comps = traj.components[reg.name]
    Δt_comps = traj.components[traj.timestep]
    for t ∈ reg.times
        zₖ = traj[t]
        vₖ = zₖ[reg.name]
        Δvₖ = vₖ - reg.baseline[:, t]
        Δtₖ = zₖ.timestep
        
        # Gradient w.r.t. variable
        ∇v = Δtₖ^2 .* (reg.R .* Δvₖ)
        v_indices = slice(t, v_comps, traj.dim)
        ∇[v_indices] .+= ∇v
        
        # Gradient w.r.t. timestep (if variable)
        if traj.timestep isa Symbol
            ∇Δt = Δvₖ' * (reg.R .* Δvₖ) * Δtₖ
            Δt_indices = slice(t, Δt_comps, traj.dim)
            ∇[Δt_indices] .+= ∇Δt
        end
    end
    
    return nothing
end

function hessian!(reg::QuadraticRegularizer, traj::NamedTrajectory)
    v_comps = traj.components[reg.name]
    Δt_comp = traj.components[traj.timestep][1]
    for t ∈ reg.times
        zₖ = traj[t]
        Δt = zₖ.timestep
        v_indices = slice(t, v_comps, traj.dim)
        Δt_index = index(t, Δt_comp, traj.dim)

        rₖ = zₖ[reg.name] - reg.baseline[:, t]

        # ∂²J/∂v² = Δt² * R
        reg.∂²J[v_indices, v_indices] = Δt^2 * spdiagm(reg.R)

        # ∂²J/∂Δt∂v = Δt * R * (v - baseline)
        reg.∂²J[v_indices, Δt_index] = 2 * Δt * reg.R .* rₖ

        # ∂²J/∂Δt² = (v - baseline)' * R * (v - baseline)
        reg.∂²J[Δt_index, Δt_index] = dot(rₖ, reg.R .* rₖ)
    end
    
    return nothing
end

function quadratic_regularizer_hessian_structure(traj::NamedTrajectory, name::Symbol, ks::Vector{Int})
    n_vars = traj.dim * traj.N + traj.global_dim
    ∂²J_structure = spzeros(n_vars, n_vars)
    v_comps = traj.components[name]
    Δt_comp = traj.components[traj.timestep][1]
    for k ∈ ks 
        v_indices = slice(k, v_comps, traj.dim)
        Δt_index = index(k, Δt_comp, traj.dim)

        # ∂²J/∂v²
        for i ∈ v_indices
            ∂²J_structure[i, i] = 1.0
        end

        # ∂²J/∂Δt∂v
        for i in v_indices
            ∂²J_structure[i, Δt_index] = 1.0
        end

        # ∂²J/∂Δt²
        ∂²J_structure[Δt_index, Δt_index] = 1.0
    end
    return ∂²J_structure 
end

function get_full_hessian(reg::QuadraticRegularizer, ::NamedTrajectory)
    return reg.∂²J
end

function hessian_structure(reg::QuadraticRegularizer, traj::NamedTrajectory)
    return quadratic_regularizer_hessian_structure(traj, reg.name, reg.times)
end

# ============================================================================ #

@testitem "testing QuadraticRegularizer" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    R = 1.0
    OBJ = QuadraticRegularizer(:u, traj, R)

    test_objective(OBJ, traj, atol=1e-5)
end
