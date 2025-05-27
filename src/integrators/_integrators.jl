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
    integrator::AbstractIntegrator; 
    show_jacobian_diff=false,
    show_hessian_diff=false,
    test_equality=true,
    atol=1e-5,
    rtol=1e-5,
    kwargs...
)

    z_dim = integrator.z_dim
    x_dim = integrator.x_dim
    u_dim = integrator.u_dim

    @test length(integrator.x_comps) == x_dim
    @test length(integrator.u_comps) == u_dim
    @test integrator.Δt_comp isa Int

    z₁ = randn(z_dim)
    z₂ = randn(z_dim)

    f̂ = zz -> begin
        δ = zeros(eltype(zz), x_dim)
        integrator(δ, zz[1:z_dim], zz[z_dim+1:end]; kwargs...)
        return δ
    end

    # testing jacobian
    ∂f = jacobian_structure(integrator)
    jacobian!(∂f, integrator, z₁, z₂)
    ∂f_autodiff = FiniteDiff.finite_difference_jacobian(f̂, [z₁; z₂])

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
    μ = randn(x_dim)
    μ∂²f = hessian_of_lagrangian(integrator, μ, z₁, z₂)
    μ∂²f_autodiff = FiniteDiff.finite_difference_hessian(zz -> μ'f̂(zz), [z₁; z₂])

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
