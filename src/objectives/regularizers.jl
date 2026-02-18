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
end

function QuadraticRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::AbstractVector{<:Real};
    baseline::AbstractMatrix{<:Real} = zeros(traj.dims[name], traj.N),
    times::AbstractVector{Int} = 1:traj.N,
)
    @assert length(R) == traj.dims[name]

    return QuadraticRegularizer(
        name,
        Vector{Float64}(R),
        Matrix{Float64}(baseline),
        Vector{Int}(times),
    )
end

function QuadraticRegularizer(name::Symbol, traj::NamedTrajectory, R::Real; kwargs...)
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

function hessian_structure(reg::QuadraticRegularizer, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    structure = spzeros(Z_dim, Z_dim)

    v_comps = traj.components[reg.name]
    Δt_comp = traj.components[traj.timestep][1]

    for k ∈ reg.times
        v_indices = slice(k, v_comps, traj.dim)
        Δt_index = index(k, Δt_comp, traj.dim)

        # ∂²J/∂v²
        structure[v_indices, v_indices] .= 1.0

        # ∂²J/∂Δt∂v
        structure[v_indices, Δt_index] .= 1.0

        # ∂²J/∂Δt²
        structure[Δt_index, Δt_index] = 1.0
    end

    return structure
end

function get_full_hessian(reg::QuadraticRegularizer, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂²J = spzeros(Z_dim, Z_dim)

    v_comps = traj.components[reg.name]
    Δt_comp = traj.components[traj.timestep][1]

    for t ∈ reg.times
        zₖ = traj[t]
        Δt = zₖ.timestep
        v_indices = slice(t, v_comps, traj.dim)
        Δt_index = index(t, Δt_comp, traj.dim)

        rₖ = zₖ[reg.name] - reg.baseline[:, t]

        # ∂²J/∂v² = Δt² * R
        ∂²J[v_indices, v_indices] = Δt^2 * spdiagm(reg.R)

        # ∂²J/∂Δt∂v = Δt * R * (v - baseline)
        ∂²J[v_indices, Δt_index] = 2 * Δt * reg.R .* rₖ

        # ∂²J/∂Δt² = (v - baseline)' * R * (v - baseline)
        ∂²J[Δt_index, Δt_index] = dot(rₖ, reg.R .* rₖ)
    end

    return ∂²J
end

# ============================================================================ #

# ----------------------------------------------------------------------------- #
# Linear Regularizer
# ----------------------------------------------------------------------------- #

"""
    LinearRegularizer <: AbstractObjective

Linear regularization objective for a trajectory component.

Computes:
```math
J = \\sum_{k \\in \\text{times}} \\sum_i R_i \\cdot v_{k,i} \\cdot \\Delta t_k
```

Used for L1 penalty via slack variables: when applied to a non-negative slack
variable `s ≥ 0` satisfying `|du| ≤ s`, minimizing `Σ R_i s_i Δt` yields
the exact L1 norm of `du`.

Gradients and Hessians are computed analytically. The Hessian has only
cross-terms ∂²J/∂v∂Δt = R_i (no diagonal).

# Fields
- `name::Symbol`: Name of the variable to regularize
- `R::Vector{Float64}`: Per-component weights
- `times::Vector{Int}`: Time indices where regularization is applied

# Constructor
```julia
LinearRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::Union{Real, AbstractVector{<:Real}};
    times::AbstractVector{Int}=1:traj.N
)
```
"""
struct LinearRegularizer <: AbstractObjective
    name::Symbol
    R::Vector{Float64}
    times::Vector{Int}
end

function LinearRegularizer(
    name::Symbol,
    traj::NamedTrajectory,
    R::AbstractVector{<:Real};
    times::AbstractVector{Int} = 1:traj.N,
)
    @assert length(R) == traj.dims[name]
    return LinearRegularizer(name, Vector{Float64}(R), Vector{Int}(times))
end

function LinearRegularizer(name::Symbol, traj::NamedTrajectory, R::Real; kwargs...)
    return LinearRegularizer(name, traj, R * ones(traj.dims[name]); kwargs...)
end

# Implement AbstractObjective interface

function objective_value(reg::LinearRegularizer, traj::NamedTrajectory)
    J = 0.0
    for t ∈ reg.times
        zₖ = traj[t]
        vₖ = zₖ[reg.name]
        Δt = zₖ.timestep
        J += Δt * dot(reg.R, vₖ)
    end
    return J
end

function gradient!(∇::AbstractVector, reg::LinearRegularizer, traj::NamedTrajectory)
    v_comps = traj.components[reg.name]
    for t ∈ reg.times
        zₖ = traj[t]
        Δtₖ = zₖ.timestep

        # ∂J/∂v_{k,i} = R_i · Δt_k
        v_indices = slice(t, v_comps, traj.dim)
        ∇[v_indices] .+= reg.R .* Δtₖ

        # ∂J/∂Δt_k = Σ_i R_i · v_{k,i}
        if traj.timestep isa Symbol
            Δt_comps = traj.components[traj.timestep]
            Δt_indices = slice(t, Δt_comps, traj.dim)
            ∇[Δt_indices] .+= dot(reg.R, zₖ[reg.name])
        end
    end
    return nothing
end

function hessian_structure(reg::LinearRegularizer, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    structure = spzeros(Z_dim, Z_dim)

    if !(traj.timestep isa Symbol)
        return structure
    end

    v_comps = traj.components[reg.name]
    Δt_comp = traj.components[traj.timestep][1]

    for k ∈ reg.times
        v_indices = slice(k, v_comps, traj.dim)
        Δt_index = index(k, Δt_comp, traj.dim)

        # Only cross-terms ∂²J/∂v∂Δt
        structure[v_indices, Δt_index] .= 1.0
    end

    return structure
end

function get_full_hessian(reg::LinearRegularizer, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂²J = spzeros(Z_dim, Z_dim)

    if !(traj.timestep isa Symbol)
        return ∂²J
    end

    v_comps = traj.components[reg.name]
    Δt_comp = traj.components[traj.timestep][1]

    for t ∈ reg.times
        v_indices = slice(t, v_comps, traj.dim)
        Δt_index = index(t, Δt_comp, traj.dim)

        # ∂²J/∂v_{k,i}∂Δt_k = R_i
        ∂²J[v_indices, Δt_index] = reg.R
    end

    return ∂²J
end

# ============================================================================ #

@testitem "testing LinearRegularizer" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    R = 1e-2
    OBJ = LinearRegularizer(:u, traj, R)

    test_objective(OBJ, traj, atol = 1e-5)
end

@testitem "testing QuadraticRegularizer" begin
    include("../../test/test_utils.jl")
    using DirectTrajOpt.Objectives

    _, traj = bilinear_dynamics_and_trajectory()

    R = 1.0
    OBJ = QuadraticRegularizer(:u, traj, R)

    test_objective(OBJ, traj, atol = 1e-5)
end
