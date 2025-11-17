module Integrators

export AbstractIntegrator
export test_integrator
export get_jacobian_structure
export get_hessian_of_lagrangian_structure

using LinearAlgebra
using SparseArrays
using ForwardDiff
using FiniteDiff
using NamedTrajectories
using TestItemRunner
using Test

# Import and extend the common interface
using ..CommonInterface
import ..CommonInterface: evaluate!, jacobian_structure, jacobian!, hessian_structure, hessian_of_lagrangian
import ..CommonInterface: eval_jacobian, eval_hessian_of_lagrangian

abstract type AbstractIntegrator end

include("derivative_integrator.jl")
include("bilinear_integrator.jl")
include("time_dependent_bilinear_integrator.jl")
include("time_integrator.jl")


include("../../test/test_utils.jl")

function get_jacobian_structure(
    integrator::AbstractIntegrator,
    traj::NamedTrajectory
)
    N = traj.N
    x_dim = integrator.x_dim
    z_dim = traj.dim
    F_dim = integrator.dim 
    Z_dim = z_dim * N + traj.global_dim
    ∂F = spzeros(F_dim, Z_dim)
    for k = 1:N-1
        ∂F[slice(k, x_dim), slice(k, 1:2z_dim, z_dim)] .= 1.0
    end 
    return ∂F
end

function get_hessian_of_lagrangian_structure(
    ::AbstractIntegrator,
    traj::NamedTrajectory
)
    N = traj.N
    z_dim = traj.dim
    Z_dim = z_dim * N + traj.global_dim
    μ∂²F = spzeros(Z_dim, Z_dim)
    for k = 1:N-1
        μ∂²F[slice(k, 1:2z_dim, z_dim), slice(k, 1:2z_dim, z_dim)] .= 1.0
    end 
    return μ∂²F
end

function test_integrator(
    integrator::AbstractIntegrator,
    traj::NamedTrajectory;
    show_jacobian_diff=false,
    show_hessian_diff=false,
    test_equality=true,
    atol=1e-5,
    rtol=1e-5
)

    # Use the provided trajectory for testing
    test_traj = traj
    
    # Constraint dimension for this trajectory
    constraint_dim = integrator.x_dim * (test_traj.N - 1)

    # Function to evaluate constraints via evaluate!
    f̂ = Z⃗ -> begin
        Z_traj = NamedTrajectory(traj; datavec=Z⃗)
        δ = zeros(eltype(Z⃗), constraint_dim)
        evaluate!(δ, integrator, Z_traj)
        return δ
    end

    # testing jacobian
    ∂f = eval_jacobian(integrator, test_traj)
    
    # Compute finite difference Jacobian
    ∂f_autodiff = FiniteDiff.finite_difference_jacobian(f̂, test_traj.datavec)

    if show_jacobian_diff 
        println("\tDifference in jacobian")
        show_diffs(∂f, ∂f_autodiff, atol=atol, rtol=rtol)
        println()
    end
    
    # Always run the tests
    if test_equality
        @test all(isapprox.(∂f, ∂f_autodiff, atol=atol, rtol=rtol))
    else
        if atol > 0.0
            @test norm(∂f - ∂f_autodiff) < atol
        else
            @test norm(∂f - ∂f_autodiff) / norm(∂f_autodiff) < rtol
        end
    end

    # testing hessian
    μ = rand(constraint_dim)
    
    μ∂²f = eval_hessian_of_lagrangian(integrator, test_traj, μ)
    
    # Compute finite difference Hessian
    μ∂²f_autodiff = FiniteDiff.finite_difference_hessian(Z⃗ -> μ'f̂(Z⃗), test_traj.datavec)

    if show_hessian_diff 
        println("\tDifference in hessian")
        show_diffs(μ∂²f, μ∂²f_autodiff, atol=atol, rtol=rtol)
        println()
    end
    
    # Always run the tests
    if test_equality
        @test all(isapprox.(triu(μ∂²f), triu(μ∂²f_autodiff), atol=atol))
    else
        if atol > 0.0
            @test norm(μ∂²f - μ∂²f_autodiff) < atol
        else
            @test norm(μ∂²f - μ∂²f_autodiff) / norm(μ∂²f_autodiff) < rtol
        end
    end

    return ∂f, ∂f_autodiff, μ∂²f, μ∂²f_autodiff
end

end
