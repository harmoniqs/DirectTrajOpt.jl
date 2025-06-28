export AbstractBilinearIntegrator
export BilinearIntegrator

using ExponentialAction
using ..Integrators

# -------------------------------------------------------------------------------- #
# Abstract Bilinear Integrator
# -------------------------------------------------------------------------------- #

abstract type AbstractBilinearIntegrator <: AbstractIntegrator end

@views function jacobian!(
    ∂f::AbstractMatrix,
    B!::AbstractBilinearIntegrator,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector,
    k::Int
)
    ForwardDiff.jacobian!(
        ∂f,
        (δ, zz) -> B!(δ, zz[1:B!.z_dim], zz[B!.z_dim+1:end], k),
        zeros(B!.x_dim),
        [zₖ; zₖ₊₁]
    )
    return nothing
end

function jacobian_structure(B::AbstractBilinearIntegrator)

    z_dim = B.z_dim
    x_dim = B.x_dim
    u_dim = B.u_dim

    x_comps = B.x_comps
    u_comps = B.u_comps
    Δt_comp = B.Δt_comp

    ∂f = spzeros(x_dim, 2 * z_dim)

    # ∂xₖ₊₁f
    ∂f[:, z_dim .+ x_comps] = I(x_dim)

    # ∂xₖf
    ∂f[:, x_comps] = ones(x_dim, x_dim)

    # ∂uₖf
    ∂f[:, u_comps] = ones(x_dim, u_dim)

    # ∂Δtₖf
    ∂f[:, Δt_comp] = ones(x_dim)

    return ∂f
end


@views function hessian_of_lagrangian(
    B!::AbstractBilinearIntegrator,
    μₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector,
    k::Int
)
    return ForwardDiff.hessian(
        zz -> begin
            δ = zeros(eltype(zz), B!.x_dim)
            B!(δ, zz[1:B!.z_dim], zz[B!.z_dim+1:end], k)
            return μₖ'δ
        end,
        [zₖ; zₖ₊₁]
    )
end

function hessian_structure(B::AbstractBilinearIntegrator)

    x_comps = B.x_comps
    u_comps = B.u_comps
    Δt_comp = B.Δt_comp

    x_dim = B.x_dim
    u_dim = B.u_dim

    μ∂²f = spzeros(2 * B.z_dim, 2 * B.z_dim)

    # μ∂ₓₖ∂ᵤf & μ∂ᵤ∂ₓₖf
    μ∂²f[x_comps, u_comps] = ones(x_dim, u_dim)

    # μ∂ₓₖ∂Δtₖf & μ∂Δtₖ∂ₓₖf
    μ∂²f[x_comps, Δt_comp] = ones(x_dim)

    # μ∂u∂Δtₖf & μ∂Δtₖ∂uf
    μ∂²f[u_comps, Δt_comp] = ones(u_dim)

    # μ∂ᵤ²f
    μ∂²f[u_comps, u_comps] = ones(u_dim, u_dim)

    # μ∂Δt²f
    μ∂²f[Δt_comp, Δt_comp] = 1.0

    return μ∂²f
end

# -------------------------------------------------------------------------------- #
# Bilinear Integrator
# -------------------------------------------------------------------------------- #

struct BilinearIntegrator{F} <: AbstractBilinearIntegrator
    G::F
    x_comps::Vector{Int}
    u_comps::Vector{Int}
    Δt_comp::Int
    z_dim::Int
    x_dim::Int
    u_dim::Int

    function BilinearIntegrator(
        G::F,
        traj::NamedTrajectory,
        xs::AbstractVector{Symbol},
        u::Symbol
    ) where F <: Function
        x_dim = sum(traj.dims[x] for x in xs)
        u_dim = traj.dims[u]

        @assert size(G(ones(u_dim))) == (x_dim, x_dim)

        x_comps = vcat([traj.components[x] for x in xs]...)
        u_comps = traj.components[u]
        Δt_comp = traj.components[traj.timestep][1]

        return new{F}(
            G,
            x_comps,
            u_comps,
            Δt_comp,
            traj.dim,
            x_dim,
            u_dim
        )
    end

    function BilinearIntegrator(G::Function, traj::NamedTrajectory, x::Symbol, u::Symbol)
        BilinearIntegrator(G, traj, [x], u)
    end
end

@views function (B::BilinearIntegrator)(
    δₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector,
    k::Int
)
    xₖ₊₁ = zₖ₊₁[B.x_comps]
    xₖ = zₖ[B.x_comps]
    uₖ = zₖ[B.u_comps]
    Δtₖ = zₖ[B.Δt_comp]
    δₖ[:] = xₖ₊₁ - expv(Δtₖ, B.G(uₖ), xₖ)
end


@testitem "testing BilinearIntegrator" begin
    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    B = BilinearIntegrator(G, traj, :x, :u)

    test_integrator(B, atol=1e-3)
end

