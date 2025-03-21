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


function dynamics_components(integrators::Vector{<:AbstractIntegrator})
    dynamics_comps = []
    comp_mark = 0
    for integrator ∈ integrators
        if(!(typeof(integrator)<:AdjointBilinearIntegrator))
            integrator_comps = comp_mark .+ (1:integrator.x_dim)
            push!(dynamics_comps, integrator_comps)
            comp_mark += integrator.x_dim
        else 
            integrator_comps = comp_mark .+ (1:integrator.x_dim+integrator.xₐ_dim)
            push!(dynamics_comps, integrator_comps)
            comp_mark += integrator.x_dim + integrator.xₐ_dim
        end
    end
    return dynamics_comps
end

function jacobian_structure(
    integrators::Vector{<:AbstractIntegrator}, 
    traj::NamedTrajectory
)
    ∂f = spzeros(0, 2traj.dim) 
    for integrator ∈ integrators
        ∂f = vcat(∂f, Integrators.jacobian_structure(integrator))
    end 
    return ∂f
end

function hessian_structure(
    integrators::Vector{<:AbstractIntegrator}, 
    traj::NamedTrajectory
)
    μ∂²f = spzeros(2traj.dim, 2traj.dim) 
    for integrator ∈ integrators
        μ∂²f .+= Integrators.hessian_structure(integrator)
    end 
    return μ∂²f
end

function get_full_hessian(μ∂²f::AbstractMatrix, traj::NamedTrajectory) 
    μ∂²F = spzeros(traj.dim * traj.T, traj.dim * traj.T)
    for k = 1:traj.T-1
        μ∂²F[slice(k, 1:2traj.dim, traj.dim), slice(k, 1:2traj.dim, traj.dim)] .+= μ∂²f
    end
    return μ∂²F
end 





"""
    TrajectoryDynamics

A struct for trajectory optimization dynamics, represented by integrators that compute
single time step dynamics, and functions for jacobians and hessians.

# Fields
- `integrators::Union{Nothing, Vector{<:AbstractIntegrator}}`: Vector of integrators.
- `F!::Function`: Function to compute trajectory dynamics.
- `∂F!::Function`: Function to compute the Jacobian of the dynamics.
- `∂fs::Vector{SparseMatrixCSC{Float64, Int}}`: Vector of Jacobian matrices.
- `μ∂²F!::Union{Function, Nothing}`: Function to compute the Hessian of the Lagrangian.
- `μ∂²fs::Vector{SparseMatrixCSC{Float64, Int}}`: Vector of Hessian matrices.
- `dim::Int`: Total dimension of the dynamics.
"""
struct TrajectoryDynamics
    F!::Function
    ∂F!::Function
    ∂fs::Vector{SparseMatrixCSC{Float64, Int}}
    μ∂²F!::Function
    μ∂²fs::Vector{SparseMatrixCSC{Float64, Int}}
    μ∂²F_structure::SparseMatrixCSC{Float64, Int}
    dim::Int

    """
        TrajectoryDynamics(
            integrators::Vector{<:AbstractIntegrator},
            traj::NamedTrajectory;
            verbose=false
        )

    Construct a `TrajectoryDynamics` object from a vector of integrators and a trajectory.
    """
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

        dynamics_dim = sum(typeof(integrator)<:AdjointBilinearIntegrator ? integrator.x_dim + integrator.xₐ_dim : integrator.x_dim  for integrator ∈ integrators) 
        dynamics_comps = dynamics_components(integrators)

        @views function F!(
            δ::AbstractVector,
            Z⃗::AbstractVector
        )
            for (integrator!, comps) ∈ zip(integrators, dynamics_comps)
                Threads.@threads for k = 1:traj.T-1
                    zₖ = Z⃗[slice(k, traj.dim)]
                    zₖ₊₁ = Z⃗[slice(k + 1, traj.dim)]
                    integrator!(δ[slice(k, comps, dynamics_dim)], zₖ, zₖ₊₁)
                end
            end
        end

        @views function ∂F!(
            ∂fs::Vector{SparseMatrixCSC{Float64, Int}},
            Z⃗::AbstractVector
        )
            for (integrator, comps) ∈ zip(integrators, dynamics_comps)
                Threads.@threads for k = 1:traj.T-1
                    zₖ = Z⃗[slice(k, traj.dim)]
                    zₖ₊₁ = Z⃗[slice(k + 1, traj.dim)]
                    jacobian!(∂fs[k][comps, :], integrator, zₖ, zₖ₊₁)
                end
            end
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
            for (integrator, comps) ∈ zip(integrators, dynamics_comps)
                Threads.@threads for k = 1:traj.T-1
                    zₖ = Z⃗[slice(k, traj.dim)]
                    zₖ₊₁ = Z⃗[slice(k + 1, traj.dim)]
                    μₖ = μ⃗[slice(k, comps, dynamics_dim)]
                    μ∂²fs[k] .+= hessian_of_lagrangian(integrator, μₖ, zₖ, zₖ₊₁)
                end
            end
        end

        ∂f_structure = jacobian_structure(integrators, traj)
        ∂fs = [copy(∂f_structure) for _ = 1:traj.T-1]

        μ∂²f_structure = hessian_structure(integrators, traj)
        μ∂²fs = [copy(μ∂²f_structure) for _ = 1:traj.T-1]

        return new(
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
    return TrajectoryDynamics(
        nothing,
        _ -> nothing,
        _ -> nothing,
        [],
        nothing,
        [],
        0
    )
end

function get_full_jacobian(D::TrajectoryDynamics, traj::NamedTrajectory) 
    ∂F = spzeros(D.dim * (traj.T - 1), traj.dim * traj.T)
    for k = 1:traj.T-1
        ∂F[slice(k, D.dim), slice(k, 1:2traj.dim, traj.dim)] += D.∂fs[k]
    end
    return ∂F
end

function get_full_hessian(D::TrajectoryDynamics, traj::NamedTrajectory) 
    μ∂²F = spzeros(traj.dim * traj.T, traj.dim * traj.T)
    for k = 1:traj.T-1
        μ∂²F[slice(k, 1:2traj.dim, traj.dim), slice(k, 1:2traj.dim, traj.dim)] .+= D.μ∂²fs[k]
    end
    return μ∂²F
end





@testitem "testing TrajectoryDynamics with single integrator" begin

    include("../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrator = BilinearIntegrator(G, traj, :x, :u)

    D = TrajectoryDynamics(integrator, traj)

    F̂ = Z⃗ -> begin
        δ = zeros(eltype(Z⃗), D.dim * (traj.T - 1))
        D.F!(δ, Z⃗)
        return δ
    end

    jacobian_autodiff = ForwardDiff.jacobian(F̂, traj.datavec)

    D.∂F!(D.∂fs, traj.datavec)

    jacobian = Dynamics.get_full_jacobian(D, traj)

    @test all(jacobian .≈ jacobian_autodiff)

    μ = randn(D.dim * (traj.T - 1))

    hessian_autodiff = ForwardDiff.hessian(Z⃗ -> μ'F̂(Z⃗), traj.datavec)

    D.μ∂²F!(D.μ∂²fs, traj.datavec, μ)

    hessian = Dynamics.get_full_hessian(D, traj)

    @test all(hessian .≈ hessian_autodiff)
end

@testitem "testing TrajectoryDynamics with multiple integrators" begin

    include("../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, traj, :x, :u),
        DerivativeIntegrator(traj, :u, :du),
        DerivativeIntegrator(traj, :du, :ddu)
    ]

    D = TrajectoryDynamics(integrators, traj)

    F̂ = Z⃗ -> begin
        δ = zeros(eltype(Z⃗), D.dim * (traj.T - 1))
        D.F!(δ, Z⃗)
        return δ
    end

    jacobian_autodiff = ForwardDiff.jacobian(F̂, traj.datavec)

    D.∂F!(D.∂fs, traj.datavec)

    @test all(Dynamics.get_full_jacobian(D, traj) .≈ jacobian_autodiff)

    μ = randn(D.dim * (traj.T - 1))

    hessian_autodiff = ForwardDiff.hessian(Z⃗ -> μ'F̂(Z⃗), traj.datavec)

    D.μ∂²F!(D.μ∂²fs, traj.datavec, μ)

    hessian = Dynamics.get_full_hessian(D, traj)

    @test all(Symmetric(hessian) .≈ hessian_autodiff)
end

end