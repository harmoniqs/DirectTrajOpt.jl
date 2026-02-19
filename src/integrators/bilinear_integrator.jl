export AbstractBilinearIntegrator
export BilinearIntegrator

using ExponentialAction
using TrajectoryIndexingUtils
using ..Integrators

# -------------------------------------------------------------------------------- #
# Abstract Bilinear Integrator
# -------------------------------------------------------------------------------- #

abstract type AbstractBilinearIntegrator <: AbstractIntegrator end

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
- `x_name::Symbol`: State variable name
- `u_name::Symbol`: Control variable name
- `x_dim::Int`: Dimension of state variable
- `var_dim::Int`: Combined dimension of all variables this integrator depends on (2*x_dim + u_dim + 1)
- `dim::Int`: Total constraint dimension (x_dim * (N-1))
- `∂fs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated compact Jacobian storage (x_dim × var_dim per timestep)
- `μ∂²fs::Vector{SparseMatrixCSC{Float64, Int}}`: Pre-allocated compact Hessian storage (var_dim × var_dim per timestep)

# Constructors
```julia
BilinearIntegrator(G::Function, x::Symbol, u::Symbol, traj::NamedTrajectory)
```

# Arguments
- `G`: Function taking control u and returning state matrix (x_dim × x_dim)
- `x`: State variable name
- `u`: Control variable name
- `traj`: Trajectory structure used to determine dimensions and pre-allocate storage

# Dynamics
Computes the constraint: x_{k+1} - exp(Δt * G(u_k)) * x_k = 0
Dependencies: xₖ, uₖ, Δtₖ, xₖ₊₁

# Example
```julia
# Linear dynamics: ẋ = (A + Σᵢ uᵢ Bᵢ) x
A = [-0.1 1.0; -1.0 -0.1]
B = [0.0 0.0; 0.0 1.0]
G = u -> A + u[1] * B

integrator = BilinearIntegrator(G, :x, :u, traj)
```
"""
struct BilinearIntegrator{F} <: AbstractBilinearIntegrator
    f::F
    x_name::Symbol
    u_name::Symbol
    x_dim::Int
    var_dim::Int
    dim::Int

    function BilinearIntegrator(G::Function, x::Symbol, u::Symbol, traj::NamedTrajectory)
        x_dim = traj.dims[x]
        u_dim = traj.dims[u]
        N = traj.N

        # Variables: [xₖ, uₖ, Δtₖ, xₖ₊₁]
        var_dim = x_dim + u_dim + 1 + x_dim  # = 2*x_dim + u_dim + 1

        # Total constraint dimension
        dim = x_dim * (N - 1)

        # Define f function: constraint is f(xₖ₊₁, xₖ, uₖ, Δtₖ) = 0
        f = (xₖ₊₁, xₖ, uₖ, Δtₖ) -> xₖ₊₁ - expv(Δtₖ, G(uₖ), xₖ)

        return new{typeof(f)}(f, x, u, x_dim, var_dim, dim)
    end
end

function Base.show(io::IO, B::BilinearIntegrator)
    print(io, "BilinearIntegrator: :$(B.x_name) = exp(Δt G(:$(B.u_name))) :$(B.x_name)  (dim = $(B.x_dim))")
end

# -------------------------------------------------------------------------------- #
# Methods
# -------------------------------------------------------------------------------- #

function evaluate!(δ::AbstractVector, B::BilinearIntegrator, traj::NamedTrajectory)
    for k = 1:(traj.N-1)
        xₖ = traj[k][B.x_name]
        xₖ₊₁ = traj[k+1][B.x_name]
        uₖ = traj[k][B.u_name]
        Δtₖ = traj[k].timestep
        δ[slice(k, B.x_dim)] = B.f(xₖ₊₁, xₖ, uₖ, Δtₖ)
    end
    return nothing
end

# Jacobian methods

@views function eval_jacobian(B::AbstractBilinearIntegrator, traj::NamedTrajectory)
    ∂B = spzeros(B.dim, traj.dim * traj.N + traj.global_dim)
    for k = 1:(traj.N-1)
        ForwardDiff.jacobian!(
            ∂B[slice(k, B.x_dim), slice(k, 1:2traj.dim, traj.dim)],
            zz -> begin
                zₖ₊₁ = zz[(traj.dim+1):end]
                zₖ = zz[1:traj.dim]

                xₖ₊₁ = zₖ₊₁[traj.components[B.x_name]]
                xₖ = zₖ[traj.components[B.x_name]]
                uₖ = zₖ[traj.components[B.u_name]]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]

                return B.f(xₖ₊₁, xₖ, uₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )
    end
    return ∂B
end

# Hessian methods

function eval_hessian_of_lagrangian(
    B::AbstractBilinearIntegrator,
    traj::NamedTrajectory,
    μ::AbstractVector,
)
    μ∂²B = spzeros(traj.dim * traj.N + traj.global_dim, traj.dim * traj.N + traj.global_dim)

    for k = 1:(traj.N-1)
        μₖ = μ[slice(k, B.x_dim)]

        μ∂²Bₖ = ForwardDiff.hessian(
            zz -> begin
                zₖ = zz[1:traj.dim]
                zₖ₊₁ = zz[(traj.dim+1):end]
                xₖ = zₖ[traj.components[B.x_name]]
                uₖ = zₖ[traj.components[B.u_name]]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]
                xₖ₊₁ = zₖ₊₁[traj.components[B.x_name]]
                return μₖ'B.f(xₖ₊₁, xₖ, uₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )

        μ∂²B[slice(k, 1:2traj.dim, traj.dim), slice(k, 1:2traj.dim, traj.dim)] .+= μ∂²Bₖ
    end
    return μ∂²B
end

# -------------------------------------------------------------------------------- #
# Tests
# -------------------------------------------------------------------------------- #

@testitem "testing BilinearIntegrator" begin
    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    B = BilinearIntegrator(G, :x, :u, traj)

    test_integrator(B, traj, atol = 1e-3)
end
