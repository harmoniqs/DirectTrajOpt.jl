export AbstractBilinearIntegrator
export BilinearIntegrator

using ExponentialAction
using ..Integrators

# -------------------------------------------------------------------------------- #
# Abstract Bilinear Integrator
# -------------------------------------------------------------------------------- #

abstract type AbstractBilinearIntegrator <: AbstractIntegrator end

@views function jacobian!(
    ∂f::AbstractMatrix,
    B!::AbstractBilinearIntegrator,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int
)
    # Extract data vectors from KnotPoints for ForwardDiff
    zₖ_vec = zₖ.data
    zₖ₊₁_vec = zₖ₊₁.data
    z_dim = length(zₖ_vec)
    
    # Get the timestep component index (assumes it's named :Δt or traj.timestep)
    # The timestep is stored in the data vector at a specific component
    timestep_indices = zₖ.components[zₖ.names[findfirst(==(:Δt), zₖ.names)]]
    timestep_idx = first(timestep_indices)
    
    ForwardDiff.jacobian!(
        ∂f,
        (δ, zz) -> begin
            # Reconstruct KnotPoints from concatenated vector for integrator call
            # Extract timestep from the data vector
            Δt_k = zz[timestep_idx]
            Δt_k₊₁ = zz[z_dim + timestep_idx]
            
            zₖ_temp = KnotPoint(
                zₖ.t, 
                view(zz, 1:z_dim), 
                Δt_k,
                zₖ.components, 
                zₖ.names, 
                zₖ.control_names
            )
            zₖ₊₁_temp = KnotPoint(
                zₖ₊₁.t, 
                view(zz, z_dim+1:2*z_dim), 
                Δt_k₊₁,
                zₖ₊₁.components, 
                zₖ₊₁.names, 
                zₖ₊₁.control_names
            )
            B!(δ, zₖ_temp, zₖ₊₁_temp, k)
        end,
        zeros(length(zₖ[B!.x_name])),
        [zₖ_vec; zₖ₊₁_vec]
    )
    return nothing
end

function jacobian_structure(B::AbstractBilinearIntegrator, traj::NamedTrajectory)

    z_dim = traj.dim
    x_dim = traj.dims[B.x_name]
    u_dim = traj.dims[B.u_name]

    x_comps = traj.components[B.x_name]
    u_comps = traj.components[B.u_name]
    Δt_comp = traj.components[traj.timestep][1]

    ∂f = spzeros(x_dim, 2 * z_dim)

    # ∂xₖ₊₁f
    ∂f[:, z_dim .+ x_comps] = I(x_dim)

    # ∂xₖf
    ∂f[:, x_comps] = ones(x_dim, x_dim)

    # ∂uₖf
    ∂f[:, u_comps] = ones(x_dim, u_dim)

    # ∂Δtₖf
    ∂f[:, Δt_comp] = ones(x_dim)

    return ∂f
end


@views function hessian_of_lagrangian(
    B!::AbstractBilinearIntegrator,
    μₖ::AbstractVector,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int
)
    # Extract data vectors from KnotPoints for ForwardDiff
    zₖ_vec = zₖ.data
    zₖ₊₁_vec = zₖ₊₁.data
    z_dim = length(zₖ_vec)
    
    # Get the timestep component index
    timestep_indices = zₖ.components[zₖ.names[findfirst(==(:Δt), zₖ.names)]]
    timestep_idx = first(timestep_indices)
    
    return ForwardDiff.hessian(
        zz -> begin
            δ = zeros(eltype(zz), length(zₖ[B!.x_name]))
            # Extract timestep from the data vector
            Δt_k = zz[timestep_idx]
            Δt_k₊₁ = zz[z_dim + timestep_idx]
            
            # Reconstruct KnotPoints from concatenated vector
            zₖ_temp = KnotPoint(
                zₖ.t, 
                view(zz, 1:z_dim), 
                Δt_k,
                zₖ.components, 
                zₖ.names, 
                zₖ.control_names
            )
            zₖ₊₁_temp = KnotPoint(
                zₖ₊₁.t, 
                view(zz, z_dim+1:2*z_dim), 
                Δt_k₊₁,
                zₖ₊₁.components, 
                zₖ₊₁.names, 
                zₖ₊₁.control_names
            )
            B!(δ, zₖ_temp, zₖ₊₁_temp, k)
            return μₖ'δ
        end,
        [zₖ_vec; zₖ₊₁_vec]
    )
end

function hessian_structure(B::AbstractBilinearIntegrator, traj::NamedTrajectory)

    x_comps = traj.components[B.x_name]
    u_comps = traj.components[B.u_name]
    Δt_comp = traj.components[traj.timestep][1]

    x_dim = traj.dims[B.x_name]
    u_dim = traj.dims[B.u_name]
    z_dim = traj.dim

    μ∂²f = spzeros(2 * z_dim, 2 * z_dim)

    # μ∂ₓₖ∂ᵤf & μ∂ᵤ∂ₓₖf
    μ∂²f[x_comps, u_comps] = ones(x_dim, u_dim)

    # μ∂ₓₖ∂Δtₖf & μ∂Δtₖ∂ₓₖf
    μ∂²f[x_comps, Δt_comp] = ones(x_dim)

    # μ∂u∂Δtₖf & μ∂Δtₖ∂uf
    μ∂²f[u_comps, Δt_comp] = ones(u_dim)

    # μ∂ᵤ²f
    μ∂²f[u_comps, u_comps] = ones(u_dim, u_dim)

    # μ∂Δt²f
    μ∂²f[Δt_comp, Δt_comp] = 1.0

    return μ∂²f
end

# -------------------------------------------------------------------------------- #
# Bilinear Integrator
# -------------------------------------------------------------------------------- #

"""
    BilinearIntegrator <: AbstractBilinearIntegrator

Integrator for control-linear dynamics of the form ẋ = G(u)x.

This integrator uses matrix exponential methods to compute accurate state transitions for
bilinear systems where the system matrix depends linearly on the control input.

# Fields
- `G::Function`: Function mapping control u to system matrix G(u)
- `x_comps::Vector{Int}`: Component indices for state variables
- `u_comps::Vector{Int}`: Component indices for control variables  
- `Δt_comp::Int`: Component index for time step
- `z_dim::Int`: Total dimension of knot point
- `x_dim::Int`: State dimension
- `u_dim::Int`: Control dimension

# Constructors
```julia
BilinearIntegrator(G::Function, x::Symbol, u::Symbol)
```

# Arguments
- `G`: Function taking control u and returning state matrix (x_dim × x_dim)
- `x`: State variable name
- `u`: Control variable name

# Dynamics
Computes the constraint: x_{k+1} - exp(Δt * G(u_k)) * x_k = 0

# Example
```julia
# Linear dynamics: ẋ = (A + Σᵢ uᵢ Bᵢ) x
A = [-0.1 1.0; -1.0 -0.1]
B = [0.0 0.0; 0.0 1.0]
G = u -> A + u[1] * B

integrator = BilinearIntegrator(G, :x, :u)
```
"""
struct BilinearIntegrator{F} <: AbstractBilinearIntegrator
    G::F
    x_name::Symbol
    u_name::Symbol

    function BilinearIntegrator(
        G::F,
        x::Symbol,
        u::Symbol
    ) where F <: Function
        return new{F}(
            G,
            x,
            u
        )
    end
end

@views function (B::BilinearIntegrator)(
    δₖ::AbstractVector,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int
)
    xₖ = zₖ[B.x_name]
    xₖ₊₁ = zₖ₊₁[B.x_name]
    uₖ = zₖ[B.u_name]
    Δtₖ = zₖ.timestep
    δₖ[:] = xₖ₊₁ - expv(Δtₖ, B.G(uₖ), xₖ)
end


@testitem "testing BilinearIntegrator" begin
    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    B = BilinearIntegrator(G, :x, :u)

    test_integrator(B, traj, atol=1e-3)
end

