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
    println("  [test_integrator] Starting...")

    # Use the provided trajectory for testing
    test_traj = traj
    
    # Constraint dimension - use integrator.dim directly
    constraint_dim = integrator.dim
    
    # Get dimensions for splitting full vector into datavec and global_data
    datavec_len = length(traj.datavec)
    global_len = traj.global_dim
    
    println("  [test_integrator] constraint_dim=$constraint_dim, datavec_len=$datavec_len, global_len=$global_len")
    
    # Full optimization vector includes both datavec and global_data
    Z⃗_full = collect(vec(traj))
    
    # Store original values for restoration
    original_datavec = copy(traj.datavec)
    original_global = global_len > 0 ? copy(traj.global_data) : nothing

    # Function to evaluate constraints via evaluate! - mutates traj in-place for efficiency
    f̂ = Z⃗ -> begin
        # Mutate trajectory data in-place
        traj.datavec .= @view Z⃗[1:datavec_len]
        if global_len > 0
            traj.global_data .= @view Z⃗[datavec_len+1:end]
        end
        δ = zeros(eltype(Z⃗), constraint_dim)
        evaluate!(δ, integrator, traj)
        return δ
    end

    println("  [test_integrator] Testing evaluate!...")
    @test !all(iszero.(f̂(Z⃗_full)))
    
    # Restore original values
    traj.datavec .= original_datavec
    if global_len > 0
        traj.global_data .= original_global
    end

    # testing jacobian
    println("  [test_integrator] Computing analytic Jacobian...")
    ∂f = eval_jacobian(integrator, test_traj)
    
    # Compute finite difference Jacobian
    println("  [test_integrator] Computing finite diff Jacobian...")
    ∂f_autodiff = FiniteDiff.finite_difference_jacobian(f̂, Z⃗_full)
    
    # Restore original values after finite diff
    traj.datavec .= original_datavec
    if global_len > 0
        traj.global_data .= original_global
    end

    if show_jacobian_diff 
        println("\tDifference in jacobian")
        show_diffs(∂f, ∂f_autodiff, atol=atol, rtol=rtol)
        println()
    end
    
    # Always run the tests
    println("  [test_integrator] Testing Jacobian...")
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
    
    println("  [test_integrator] Computing analytic Hessian...")
    μ∂²f = eval_hessian_of_lagrangian(integrator, test_traj, μ)
    
    # Compute finite difference Hessian
    println("  [test_integrator] Computing finite diff Hessian...")
    μ∂²f_autodiff = FiniteDiff.finite_difference_hessian(Z⃗ -> μ'f̂(Z⃗), Z⃗_full)
    
    # Restore original values after finite diff
    traj.datavec .= original_datavec
    if global_len > 0
        traj.global_data .= original_global
    end

    if show_hessian_diff 
        println("\tDifference in hessian")
        show_diffs(μ∂²f, μ∂²f_autodiff, atol=atol, rtol=rtol)
        println()
    end
    
    # Always run the tests
    println("  [test_integrator] Testing Hessian...")
    if test_equality
        @test all(isapprox.(triu(μ∂²f), triu(μ∂²f_autodiff), atol=atol))
    else
        if atol > 0.0
            @test norm(μ∂²f - μ∂²f_autodiff) < atol
        else
            @test norm(μ∂²f - μ∂²f_autodiff) / norm(μ∂²f_autodiff) < rtol
        end
    end

    println("  [test_integrator] Done!")
    return ∂f, ∂f_autodiff, μ∂²f, μ∂²f_autodiff
end

end
