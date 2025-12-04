export DerivativeIntegrator

"""
    DerivativeIntegrator <: AbstractIntegrator

Integrator for derivative constraints of the form xₖ₊₁ - xₖ - Δt * ẋₖ = 0.

This enforces smoothness by relating a variable to its derivative.

# Fields
- `f::Function`: Constraint function f(xₖ₊₁, xₖ, ẋₖ, Δtₖ) = xₖ₊₁ - xₖ - Δtₖ * ẋₖ
- `x_name::Symbol`: Variable name
- `ẋ_name::Symbol`: Derivative variable name
- `x_dim::Int`: Dimension of variable
- `var_dim::Int`: Combined dimension (2*x_dim + 1 for xₖ, ẋₖ, Δtₖ, xₖ₊₁)
- `dim::Int`: Total constraint dimension (x_dim * (N-1))
- `∂fs::Vector{SparseMatrixCSC{Float64, Int}}`: Compact Jacobian storage
- `μ∂²fs::Vector{SparseMatrixCSC{Float64, Int}}`: Compact Hessian storage

# Example
```julia
# Enforce velocity smoothness: vₖ₊₁ - vₖ - Δt * aₖ = 0
integrator = DerivativeIntegrator(:v, :a, traj)
```
"""
struct DerivativeIntegrator{F} <: AbstractIntegrator
    f::F
    x_name::Symbol
    ẋ_name::Symbol
    x_dim::Int
    var_dim::Int
    dim::Int

    function DerivativeIntegrator(
        x::Symbol,
        ẋ::Symbol,
        traj::NamedTrajectory
    )
        x_dim = traj.dims[x]
        N = traj.N
        
        # Variables: [xₖ, ẋₖ, Δtₖ, xₖ₊₁]
        var_dim = 2*x_dim + 1 + x_dim  # = 3*x_dim + 1
        
        # Total constraint dimension
        dim = x_dim * (N - 1)
        
        # Define f function: constraint is f(xₖ₊₁, xₖ, ẋₖ, Δtₖ) = 0
        f = (xₖ₊₁, xₖ, ẋₖ, Δtₖ) -> xₖ₊₁ - xₖ - Δtₖ * ẋₖ
        
        return new{typeof(f)}(
            f,
            x,
            ẋ,
            x_dim,
            var_dim,
            dim
        )
    end
end

function evaluate!(
    δ::AbstractVector,
    D::DerivativeIntegrator,
    traj::NamedTrajectory,
)
    for k = 1:traj.N-1
        xₖ = traj[k][D.x_name]
        xₖ₊₁ = traj[k+1][D.x_name]
        ẋₖ = traj[k][D.ẋ_name]
        Δtₖ = traj[k].timestep
        δ[slice(k, D.x_dim)] = D.f(xₖ₊₁, xₖ, ẋₖ, Δtₖ)
    end
    return nothing
end

# Jacobian methods

@views function eval_jacobian(
    D::DerivativeIntegrator,
    traj::NamedTrajectory
)
    ∂D = spzeros(D.dim, traj.dim * traj.N + traj.global_dim)
    for k = 1:traj.N-1
        ForwardDiff.jacobian!(
            ∂D[slice(k, D.x_dim), slice(k, 1:2traj.dim, traj.dim)],
            zz -> begin 
                zₖ = zz[1:traj.dim]
                zₖ₊₁ = zz[traj.dim+1:end]
                xₖ = zₖ[traj.components[D.x_name]]
                ẋₖ = zₖ[traj.components[D.ẋ_name]]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]
                xₖ₊₁ = zₖ₊₁[traj.components[D.x_name]]
                return D.f(xₖ₊₁, xₖ, ẋₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )
    end
    return ∂D 
end

# Hessian methods

function eval_hessian_of_lagrangian(
    D::DerivativeIntegrator,
    traj::NamedTrajectory,
    μ::AbstractVector
)
    μ∂²D = spzeros(
        traj.dim * traj.N + traj.global_dim,
        traj.dim * traj.N + traj.global_dim,
    )

    for k = 1:traj.N-1
        μₖ = μ[slice(k, D.x_dim)]
       
        μ∂²Dₖ = ForwardDiff.hessian(
            zz -> begin
                zₖ = zz[1:traj.dim]
                zₖ₊₁ = zz[traj.dim+1:end]
                xₖ = zₖ[traj.components[D.x_name]]
                ẋₖ = zₖ[traj.components[D.ẋ_name]]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]
                xₖ₊₁ = zₖ₊₁[traj.components[D.x_name]]
                return μₖ'D.f(xₖ₊₁, xₖ, ẋₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )

        μ∂²D[slice(k, 1:2traj.dim, traj.dim), slice(k, 1:2traj.dim, traj.dim)] .+= μ∂²Dₖ
    end
    return μ∂²D 
end

@testitem "testing DerivativeIntegrator" begin
    include("../../test/test_utils.jl")
    traj = named_trajectory_type_1()
    D = DerivativeIntegrator(:a, :da, traj)
    test_integrator(D, traj)
end
