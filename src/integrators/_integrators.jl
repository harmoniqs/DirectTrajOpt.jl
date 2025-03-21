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
include("adjoint_bilinear_integrator.jl")


include("../../test/test_utils.jl")

function test_integrator(integrator::AbstractIntegrator; diff=false)

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
        integrator(δ, zz[1:z_dim], zz[z_dim+1:end])
        return δ
    end

    # testing jacobian
    ∂f = jacobian_structure(integrator)

    jacobian!(∂f, integrator, z₁, z₂)

    ∂f_autodiff = FiniteDiff.finite_difference_jacobian(f̂, [z₁; z₂])

    @test all(isapprox.(∂f, ∂f_autodiff, atol=1e-5))
    if diff 
        show_diffs(∂f, ∂f_autodiff)
    end

    # testing hessian
    μ = randn(x_dim)

    μ∂²f = hessian_of_lagrangian(integrator, μ, z₁, z₂)

    μ∂²f_autodiff = FiniteDiff.finite_difference_hessian(zz -> μ'f̂(zz), [z₁; z₂])

    @test all(isapprox.(Symmetric(μ∂²f), μ∂²f_autodiff, atol=1e-5))

    if diff 
        display(μ∂²f)
        display(sparse(μ∂²f_autodiff))
        show_diffs(μ∂²f, μ∂²f_autodiff)
    end
end

end
