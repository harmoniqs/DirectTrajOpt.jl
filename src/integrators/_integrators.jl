module Integrators

export AbstractIntegrator
export jacobian_structure
export jacobian!
export hessian_structure
export hessian_of_lagrangian
export test_integrator

using LinearAlgebra
using SparseArrays
using ForwardDiff
using FiniteDiff
using NamedTrajectories
using TestItemRunner
using Test

abstract type AbstractIntegrator end

include("derivative_integrator.jl")
include("bilinear_integrator.jl")
include("time_dependent_bilinear_integrator.jl")
include("time_integrator.jl")


include("../../test/test_utils.jl")

function test_integrator(
    integrator::AbstractIntegrator,
    traj::NamedTrajectory;
    show_jacobian_diff=false,
    show_hessian_diff=false,
    test_equality=true,
    atol=1e-5,
    rtol=1e-5
)

    z_dim = traj.dim
    
    # Get the state dimension from the trajectory using the integrator's x_name or t_name
    if hasfield(typeof(integrator), :x_name)
        x_dim = traj.dims[integrator.x_name]
    elseif hasfield(typeof(integrator), :t_name)
        x_dim = traj.dims[integrator.t_name]
    else
        error("Integrator type $(typeof(integrator)) must have either x_name or t_name field")
    end

    # Use the trajectory's component structure to create KnotPoints
    z₁_vec = randn(z_dim)
    z₂_vec = randn(z_dim)
    k = 1

    # Get timestep index
    timestep_comp = traj.components[traj.timestep][1]

    # Create KnotPoints using trajectory structure
    z₁ = KnotPoint(
        1, 
        z₁_vec, 
        z₁_vec[timestep_comp],
        traj.components,
        traj.names,
        traj.control_names
    )
    z₂ = KnotPoint(
        2, 
        z₂_vec, 
        z₂_vec[timestep_comp],
        traj.components,
        traj.names,
        traj.control_names
    )

    f̂ = zz -> begin
        δ = zeros(eltype(zz), x_dim)
        # Create temporary KnotPoints from the concatenated vector
        z₁_temp = KnotPoint(1, view(zz, 1:z_dim), zz[timestep_comp], traj.components, traj.names, traj.control_names)
        z₂_temp = KnotPoint(2, view(zz, z_dim+1:2*z_dim), zz[z_dim + timestep_comp], traj.components, traj.names, traj.control_names)
        integrator(δ, z₁_temp, z₂_temp, k)
        return δ
    end

    # testing jacobian
    ∂f = jacobian_structure(integrator, traj)
    jacobian!(∂f, integrator, z₁, z₂, k)
    ∂f_autodiff = FiniteDiff.finite_difference_jacobian(f̂, [z₁_vec; z₂_vec])

    if show_jacobian_diff 
        println("\tDifference in jacobian")
        show_diffs(∂f, ∂f_autodiff, atol=atol, rtol=rtol)
        println()
    else
        if test_equality
            @test all(isapprox.(∂f, ∂f_autodiff, atol=atol, rtol=rtol))
        else
            if atol > 0.0
                @test norm(∂f - ∂f_autodiff) < atol
            else
                @test norm(∂f - ∂f_autodiff) / norm(∂f_autodiff) < rtol
            end
        end
    end

    # testing hessian
    μ = rand(x_dim)
    μ∂²f = hessian_of_lagrangian(integrator, μ, z₁, z₂, k)
    μ∂²f_autodiff = FiniteDiff.finite_difference_hessian(zz -> μ'f̂(zz), [z₁_vec; z₂_vec])

    if show_hessian_diff 
        println("\tDifference in hessian")
        show_diffs(μ∂²f, μ∂²f_autodiff, atol=atol, rtol=rtol)
        println()
    else
        if test_equality
            @test all(isapprox.(Symmetric(μ∂²f), μ∂²f_autodiff, atol=atol))
        else
            if atol > 0.0
                @test norm(μ∂²f - μ∂²f_autodiff) < atol
            else
                @test norm(μ∂²f - μ∂²f_autodiff) / norm(μ∂²f_autodiff) < rtol
            end
        end
    end

    return ∂f, ∂f_autodiff, μ∂²f, μ∂²f_autodiff
end

end
