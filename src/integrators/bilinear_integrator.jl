export BilinearIntegrator

using ExponentialAction

struct BilinearIntegrator <: AbstractIntegrator
    G::Function
    x_comps::AbstractVector{Int}
    u_comps::AbstractVector{Int}
    Δt_comp::Int
    z_dim::Int
    x_dim::Int
    u_dim::Int

    function BilinearIntegrator(
        G::Function,
        traj::NamedTrajectory,
        x::Symbol,
        u::Symbol,
        Δt::Symbol
    )
        @assert size(G(traj[1][u])) == (traj.dims[x], traj.dims[x]) 

        return new(
            G,
            traj.components[x],
            traj.components[u],
            traj.components[Δt][1],
            traj.dim,
            traj.dims[x],
            traj.dims[u]
        )
    end
end

function (B::BilinearIntegrator)(
    δₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    xₖ₊₁ = zₖ₊₁[B.x_comps]
    xₖ = zₖ[B.x_comps]
    uₖ = zₖ[B.u_comps]
    Δtₖ = zₖ[B.Δt_comp]
    δₖ[:] = xₖ₊₁ - expv(Δtₖ, B.G(uₖ), xₖ)
end

function jacobian!(
    ∂f::AbstractMatrix,
    B!::BilinearIntegrator,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    ForwardDiff.jacobian!(
        ∂f,
        (δ, zz) -> B!(δ, zz[1:B!.z_dim], zz[B!.z_dim+1:end]),
        zeros(B!.x_dim),
        [zₖ; zₖ₊₁]
    )
end

function jacobian_structure(B::BilinearIntegrator)

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


function hessian_of_lagrangian(
    B!::BilinearIntegrator,
    μₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    return ForwardDiff.hessian(
        zz -> begin
            δ = zeros(eltype(zz), B!.x_dim)
            B!(δ, zz[1:B!.z_dim], zz[B!.z_dim+1:end])
            return μₖ'δ
        end,
        [zₖ; zₖ₊₁]
    )
end

function hessian_structure(B::BilinearIntegrator)

    x_comps = B.x_comps
    u_comps = B.u_comps
    Δt_comp = B.Δt_comp

    x_dim = B.x_dim
    u_dim = B.u_dim

    μ∂²f = spzeros(2 * B.z_dim, 2 * B.z_dim)

    # μ∂ₓₖ∂ᵤf & μ∂ᵤ∂ₓₖf
    μ∂²f[x_comps, u_comps] = ones(x_dim, u_dim)
    μ∂²f[u_comps, x_comps] = ones(u_dim, x_dim)

    # μ∂ₓₖ∂Δtₖf & μ∂Δtₖ∂ₓₖf
    μ∂²f[x_comps, Δt_comp] = ones(x_dim)
    μ∂²f[Δt_comp, x_comps] = ones(x_dim)

    # μ∂u∂Δtₖf & μ∂Δtₖ∂uf
    μ∂²f[u_comps, Δt_comp] = ones(u_dim)
    μ∂²f[Δt_comp, u_comps] = ones(u_dim)

    # μ∂ᵤ²f
    μ∂²f[u_comps, u_comps] = ones(u_dim, u_dim)

    # μ∂Δt²f
    μ∂²f[Δt_comp, Δt_comp] = 1.0

    return μ∂²f
end



@testitem "testing BilinearIntegrator" begin
    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    B = BilinearIntegrator(G, traj, :x, :u, :Δt)

    test_integrator(B; diff=false)
end

