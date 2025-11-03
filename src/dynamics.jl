module Dynamics

export AbstractDynamics
export TrajectoryDynamics

export dynamics
export dynamics_jacobian
export dynamics_hessian_of_lagrangian
export dynamics_components

using ..Integrators

using TrajectoryIndexingUtils
using NamedTrajectories
using LinearAlgebra
using SparseArrays
using ForwardDiff
using TestItemRunner


function dynamics_components(integrators::Vector{<:AbstractIntegrator}, traj::NamedTrajectory)
    dynamics_comps = []
    comp_mark = 0
    for integrator ∈ integrators
        # Get the state dimension from the trajectory using the integrator's x_name or t_name
        if hasfield(typeof(integrator), :x_name)
            x_dim = traj.dims[integrator.x_name]
        elseif hasfield(typeof(integrator), :t_name)
            x_dim = traj.dims[integrator.t_name]
        else
            error("Integrator type $(typeof(integrator)) must have either x_name or t_name field")
        end
        integrator_comps = comp_mark .+ (1:x_dim)
        push!(dynamics_comps, integrator_comps)
        comp_mark += x_dim
    end
    return dynamics_comps
end

function jacobian_structure(
    integrators::Vector{<:AbstractIntegrator}, 
    traj::NamedTrajectory
)
    ∂f = spzeros(0, 2traj.dim) 
    for integrator ∈ integrators
        ∂f = vcat(∂f, Integrators.jacobian_structure(integrator, traj))
    end 
    return ∂f
end

function hessian_structure(
    integrators::Vector{<:AbstractIntegrator}, 
    traj::NamedTrajectory
)
    μ∂²f = spzeros(2traj.dim, 2traj.dim) 
    for integrator ∈ integrators
        μ∂²f .+= Integrators.hessian_structure(integrator, traj)
    end 
    return μ∂²f
end

function get_full_hessian(μ∂²f::AbstractMatrix, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    μ∂²F = spzeros(Z_dim, Z_dim)
    for k = 1:traj.N-1
        μ∂²F[slice(k, 1:2traj.dim, traj.dim), slice(k, 1:2traj.dim, traj.dim)] .+= μ∂²f
    end
    return μ∂²F
end 



"""
    TrajectoryDynamics

Represents the dynamics of a trajectory optimization problem through integrators.

This structure encapsulates the dynamics constraints that enforce consistency between
consecutive time steps in the trajectory. It uses integrators to define how the state
evolves from one time step to the next and provides automatic differentiation support
through Jacobian and Hessian computations.

# Fields
- `trajectory::NamedTrajectory`: The trajectory structure used to wrap datavectors
- `F!::Function`: In-place function computing dynamics violations δ = f(zₖ, zₖ₊₁)
- `∂F!::Function`: In-place function computing Jacobian of dynamics
- `∂fs::Vector{SparseMatrixCSC}`: Cached Jacobian matrices for each time step
- `μ∂²F!::Function`: In-place function computing Hessian of Lagrangian
- `μ∂²fs::Vector{SparseMatrixCSC}`: Cached Hessian matrices for each time step
- `μ∂²F_structure::SparseMatrixCSC`: Sparsity structure of full trajectory Hessian
- `dim::Int`: Total dimension of dynamics (sum of all integrator state dimensions)

# Constructor
```julia
TrajectoryDynamics(
    integrators::Vector{<:AbstractIntegrator},
    traj::NamedTrajectory;
    verbose=false
)
```

Create trajectory dynamics from integrators and a trajectory structure.

# Example
```julia
G = rand(2, 2)
integrator = BilinearIntegrator(G, :x, :u)
dynamics = TrajectoryDynamics(integrator, traj)
```
"""
struct TrajectoryDynamics{F1, F2, F3} 
    trajectory::NamedTrajectory
    F!::F1
    ∂F!::F2
    ∂fs::Vector{SparseMatrixCSC{Float64, Int}}
    μ∂²F!::F3
    μ∂²fs::Vector{SparseMatrixCSC{Float64, Int}}
    μ∂²F_structure::SparseMatrixCSC{Float64, Int}
    dim::Int

    function TrajectoryDynamics(
        integrators::Vector{<:AbstractIntegrator},
        traj::NamedTrajectory;
        verbose=false
    )
        if length(integrators) == 0
            if verbose
                println("        constructing Null dynamics function...")
            end
            return NullTrajectoryDynamics()
        end

        if verbose
            println("        constructing full dynamics derivative functions...")
        end

        dynamics_comps = dynamics_components(integrators, traj)
        
        # Calculate total dynamics dimension from components
        dynamics_dim = sum(length(comps) for comps in dynamics_comps)

        @views function F!(
            δ::AbstractVector,
            Z⃗::AbstractVector
        )
            # Wrap datavector in NamedTrajectory to access KnotPoints
            Z_traj = NamedTrajectory(traj; datavec=Z⃗)
            for (integrator!, comps) ∈ zip(integrators, dynamics_comps)
                Threads.@threads for k = 1:traj.N-1
                    zₖ = Z_traj[k]
                    zₖ₊₁ = Z_traj[k + 1]
                    integrator!(δ[slice(k, comps, dynamics_dim)], zₖ, zₖ₊₁, k)
                end
            end
            return nothing
        end

        @views function ∂F!(
            ∂fs::Vector{SparseMatrixCSC{Float64, Int}},
            Z⃗::AbstractVector
        )
            # Wrap datavector in NamedTrajectory to access KnotPoints
            Z_traj = NamedTrajectory(traj; datavec=Z⃗)
            for (integrator, comps) ∈ zip(integrators, dynamics_comps)
                Threads.@threads for k = 1:traj.N-1
                    zₖ = Z_traj[k]
                    zₖ₊₁ = Z_traj[k + 1]
                    jacobian!(∂fs[k][comps, :], integrator, zₖ, zₖ₊₁, k)
                end
            end
            return nothing
        end

        @views function μ∂²F!(
            μ∂²fs::Vector{SparseMatrixCSC{Float64, Int}},
            Z⃗::AbstractVector,
            μ⃗::AbstractVector
        )
            # TODO: figure out how to get around resetting the matrix to zero
            for μ∂²f ∈ μ∂²fs
                μ∂²f .= 0.0
            end
            # Wrap datavector in NamedTrajectory to access KnotPoints
            Z_traj = NamedTrajectory(traj; datavec=Z⃗)
            for (integrator, comps) ∈ zip(integrators, dynamics_comps)
                Threads.@threads for k = 1:traj.N-1
                    zₖ = Z_traj[k]
                    zₖ₊₁ = Z_traj[k + 1]
                    μₖ = μ⃗[slice(k, comps, dynamics_dim)]
                    μ∂²fs[k] .+= hessian_of_lagrangian(integrator, μₖ, zₖ, zₖ₊₁, k)
                end
            end
            return nothing
        end

        ∂f_structure = jacobian_structure(integrators, traj)
        ∂fs = [copy(∂f_structure) for _ = 1:traj.N-1]

        μ∂²f_structure = hessian_structure(integrators, traj)
        μ∂²fs = [copy(μ∂²f_structure) for _ = 1:traj.N-1]

        return new{typeof(F!), typeof(∂F!), typeof(μ∂²F!)}(
            traj,
            F!,
            ∂F!,
            ∂fs,
            μ∂²F!,
            μ∂²fs,
            get_full_hessian(μ∂²f_structure, traj),
            dynamics_dim
        )
    end
end


function TrajectoryDynamics(
    integrator::AbstractIntegrator,
    traj::NamedTrajectory;
    kwargs...
)
    return TrajectoryDynamics([integrator], traj; kwargs...)
end

function NullTrajectoryDynamics()
    # Create a minimal dummy trajectory
    dummy_traj = NamedTrajectory((x = zeros(1, 1),), timestep=:Δt)
    return TrajectoryDynamics(
        dummy_traj,
        (_, _) -> nothing,
        (_, _) -> nothing,
        SparseMatrixCSC{Float64, Int}[],
        (_, _, _) -> nothing,
        SparseMatrixCSC{Float64, Int}[],
        spzeros(0, 0),
        0
    )
end

function get_full_jacobian(D::TrajectoryDynamics, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    ∂F = spzeros(D.dim * (traj.N - 1), Z_dim)
    for k = 1:traj.N-1
        ∂F[slice(k, D.dim), slice(k, 1:2traj.dim, traj.dim)] += D.∂fs[k]
    end
    return ∂F
end

function get_full_hessian(D::TrajectoryDynamics, traj::NamedTrajectory)
    Z_dim = traj.dim * traj.N + traj.global_dim
    μ∂²F = spzeros(Z_dim, Z_dim)
    for k = 1:traj.N-1
        μ∂²F[slice(k, 1:2traj.dim, traj.dim), slice(k, 1:2traj.dim, traj.dim)] .+= D.μ∂²fs[k]
    end
    return μ∂²F
end





@testitem "testing TrajectoryDynamics with single integrator" begin
    using FiniteDiff

    include("../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrator = BilinearIntegrator(G, :x, :u)

    D = TrajectoryDynamics(integrator, traj)

    F̂ = Z⃗ -> begin
        δ = zeros(eltype(Z⃗), D.dim * (traj.N - 1))
        D.F!(δ, Z⃗)
        return δ
    end

    jacobian_finitediff = FiniteDiff.finite_difference_jacobian(F̂, traj.datavec)

    D.∂F!(D.∂fs, traj.datavec)

    jacobian = Dynamics.get_full_jacobian(D, traj)

    @test all(isapprox.(jacobian, jacobian_finitediff, atol=1e-6, rtol=1e-6))

    μ = randn(D.dim * (traj.N - 1))

    hessian_finitediff = FiniteDiff.finite_difference_hessian(Z⃗ -> μ'F̂(Z⃗), traj.datavec)

    D.μ∂²F!(D.μ∂²fs, traj.datavec, μ)

    hessian = Dynamics.get_full_hessian(D, traj)

    @test all(isapprox.(hessian, hessian_finitediff, atol=1e-6, rtol=1e-6))
end

@testitem "testing TrajectoryDynamics with multiple integrators" begin
    using FiniteDiff

    include("../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u),
        DerivativeIntegrator(:u, :du),
        DerivativeIntegrator(:du, :ddu)
    ]

    D = TrajectoryDynamics(integrators, traj)

    F̂ = Z⃗ -> begin
        δ = zeros(eltype(Z⃗), D.dim * (traj.N - 1))
        D.F!(δ, Z⃗)
        return δ
    end

    jacobian_finitediff = FiniteDiff.finite_difference_jacobian(F̂, traj.datavec)

    D.∂F!(D.∂fs, traj.datavec)

    @test all(isapprox.(Dynamics.get_full_jacobian(D, traj), jacobian_finitediff, atol=1e-6, rtol=1e-6))

    μ = randn(D.dim * (traj.N - 1))

    hessian_finitediff = FiniteDiff.finite_difference_hessian(Z⃗ -> μ'F̂(Z⃗), traj.datavec)

    D.μ∂²F!(D.μ∂²fs, traj.datavec, μ)

    hessian = Dynamics.get_full_hessian(D, traj)

    @test all(isapprox.(Symmetric(hessian), hessian_finitediff, atol=1e-6, rtol=1e-6))
end

end