export TimeIntegrator

"""
    TimeIntegrator <: AbstractIntegrator

Integrator for time progression constraint: tₖ₊₁ - tₖ - Δtₖ = 0.

This enforces that time progresses consistently with timesteps.

# Fields
- `f::Function`: Constraint function f(tₖ₊₁, tₖ, Δtₖ) = tₖ₊₁ - tₖ - Δtₖ
- `t_name::Symbol`: Time variable name  
- `x_dim::Int`: Dimension (always 1 for scalar time)
- `var_dim::Int`: Combined dimension (1 + 1 + 1 = 3 for tₖ, Δtₖ, tₖ₊₁)
- `dim::Int`: Total constraint dimension (1 * (N-1))
- `∂fs::Vector{SparseMatrixCSC{Float64, Int}}`: Compact Jacobian storage
- `μ∂²fs::Vector{SparseMatrixCSC{Float64, Int}}`: Compact Hessian storage (zeros, linear constraint)

# Example
```julia
# Enforce time progression
integrator = TimeIntegrator(:t, traj)
```
"""
struct TimeIntegrator{F} <: AbstractIntegrator
    f::F
    t_name::Symbol
    x_dim::Int
    var_dim::Int
    dim::Int

    function TimeIntegrator(
        t::Symbol,
        traj::NamedTrajectory
    )
        x_dim = 1  # Time is scalar
        N = traj.N
        
        # Variables: [tₖ, Δtₖ, tₖ₊₁]
        var_dim = 3
        
        # Total constraint dimension
        dim = N - 1
        
        # Define f function: constraint is f(tₖ₊₁, tₖ, Δtₖ) = 0
        f = (tₖ₊₁, tₖ, Δtₖ) -> tₖ₊₁ - tₖ - Δtₖ
        
        return new{typeof(f)}(
            f,
            t,
            x_dim,
            var_dim,
            dim
        )
    end
end

function evaluate!(
    δ::AbstractVector,
    T::TimeIntegrator,
    traj::NamedTrajectory,
)
    for k = 1:traj.N-1
        tₖ = traj[k][T.t_name][1]
        tₖ₊₁ = traj[k+1][T.t_name][1]
        Δtₖ = traj[k].timestep
        δ[k] = T.f(tₖ₊₁, tₖ, Δtₖ)
    end
    return nothing
end

# Jacobian methods

@views function eval_jacobian(
    T::TimeIntegrator,
    traj::NamedTrajectory
)
    ∂T = spzeros(T.dim, traj.dim * traj.N + traj.global_dim)
    for k = 1:traj.N-1
        ForwardDiff.jacobian!(
            ∂T[k:k, slice(k, 1:2traj.dim, traj.dim)],
            zz -> begin 
                zₖ = zz[1:traj.dim]
                zₖ₊₁ = zz[traj.dim+1:end]
                tₖ = zₖ[traj.components[T.t_name]][1]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]
                tₖ₊₁ = zₖ₊₁[traj.components[T.t_name]][1]
                return [T.f(tₖ₊₁, tₖ, Δtₖ)]
            end,
            [traj[k].data; traj[k+1].data],
        )
    end
    return ∂T 
end

# Hessian methods

function eval_hessian_of_lagrangian(
    T::TimeIntegrator,
    traj::NamedTrajectory,
    μ::AbstractVector
)
    # Hessian is zero for linear constraint
    μ∂²T = spzeros(
        traj.dim * traj.N + traj.global_dim,
        traj.dim * traj.N + traj.global_dim,
    )
    return μ∂²T 
end
