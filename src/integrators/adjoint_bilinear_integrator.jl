
export AdjointBilinearIntegrator

using ExponentialAction

struct AdjointBilinearIntegrator <: AbstractIntegrator
    G::Function
    x_comps::AbstractVector{Int}
    xₐ_comps::AbstractVector{Int}
    u_comps::AbstractVector{Int}
    Δt_comp::Int
    z_dim::Int
    x_dim::Int
    xₐ_dim::Int
    u_dim::Int
    

    function AdjointBilinearIntegrator(
        G::Function,
        traj::NamedTrajectory,
        x::Symbol,
        xₐ::Symbol,
        u::Symbol
    )
        #TODO put the right assert here
        #@assert size(G(traj[1][u])) == (traj.dims[x], traj.dims[x])  

        return new(
            G,
            traj.components[x],
            traj.components[xₐ],
            traj.components[u],
            traj.components[traj.timestep][1],
            traj.dim,
            traj.dims[x],
            traj.dims[xₐ],
            traj.dims[u]
        )
    end
end

@views function (B::AdjointBilinearIntegrator)(
    δₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    xₖ₊₁ = zₖ₊₁[B.x_comps]
    xₖ = zₖ[B.x_comps]

    xₐₖ₊₁ = zₖ₊₁[B.xₐ_comps]
    xₐₖ = zₖ[B.xₐ_comps]
    
    uₖ = zₖ[B.u_comps]
    Δtₖ = zₖ[B.Δt_comp]
    δₖ[:] = vcat(xₐₖ₊₁,xₖ₊₁) - expv(Δtₖ, B.G(uₖ), vcat(xₐₖ,xₖ))
end

@views function jacobian!(
    ∂f::AbstractMatrix,
    B!::AdjointBilinearIntegrator,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    ForwardDiff.jacobian!(
        ∂f,
        (δ, zz) -> B!(δ, zz[1:B!.z_dim], zz[B!.z_dim+1:end]),
        zeros(B!.x_dim+B!.xₐ_dim),
        [zₖ; zₖ₊₁]
    )
end

function jacobian_structure(B::AdjointBilinearIntegrator)

    z_dim = B.z_dim
    x_dim = B.x_dim
    xₐ_dim = B.x_dim
    u_dim = B.u_dim
    

    x_comps = B.x_comps
    xₐ_comps = B.xₐ_comps
    u_comps = B.u_comps
    Δt_comp = B.Δt_comp

    ∂f = spzeros(x_dim+xₐ_dim, 2 * z_dim)

    # ∂xₖ₊₁f
    ∂f[1:x_dim, z_dim .+ x_comps] = I(x_dim)

    # ∂xₖf
    ∂f[1:x_dim, x_comps] = ones(x_dim, x_dim)

    # ∂xₖ₊₁f
    ∂f[x_dim+1:x_dim+xₐ_dim, z_dim .+ xₐ_comps] = I(xₐ_dim)

    # ∂xₖf
    ∂f[x_dim+1:x_dim+xₐ_dim,  xₐ_comps] = ones(xₐ_dim, xₐ_dim)

    # ∂uₖf
    ∂f[:, u_comps] = ones(x_dim+xₐ_dim, u_dim)

    # ∂Δtₖf
    ∂f[:, Δt_comp] = ones(x_dim+xₐ_dim)

    return ∂f
end


@views function hessian_of_lagrangian(
    B!::AdjointBilinearIntegrator,
    μₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    return ForwardDiff.hessian(
        zz -> begin
            δ = zeros(eltype(zz), B!.x_dim+B!.xₐ_dim)
            B!(δ, zz[1:B!.z_dim], zz[B!.z_dim+1:end])
            return μₖ'δ
        end,
        [zₖ; zₖ₊₁]
    )
end

function hessian_structure(B::AdjointBilinearIntegrator)

    x_comps = B.x_comps
    xₐ_comps = B.xₐ_comps
    u_comps = B.u_comps
    Δt_comp = B.Δt_comp

    x_dim = B.x_dim
    xₐ_dim = B.xₐ_dim
    u_dim = B.u_dim

    μ∂²f = spzeros(2 * B.z_dim, 2 * B.z_dim)

    # μ∂ₓₖ∂ᵤf & μ∂ᵤ∂ₓₖf
    μ∂²f[x_comps, u_comps] = ones(x_dim, u_dim)

    # μ∂ₓₖ∂Δtₖf & μ∂Δtₖ∂ₓₖf
    μ∂²f[x_comps, Δt_comp] = ones(x_dim)

    # μ∂ₓₖ∂ᵤf & μ∂ᵤ∂ₓₖf
    μ∂²f[xₐ_comps, u_comps] = ones(xₐ_dim, u_dim)

    # μ∂ₓₖ∂Δtₖf & μ∂Δtₖ∂ₓₖf
    μ∂²f[xₐ_comps, Δt_comp] = ones(xₐ_dim)

    # μ∂u∂Δtₖf & μ∂Δtₖ∂uf
    μ∂²f[u_comps, Δt_comp] = ones(u_dim)

    # μ∂ᵤ²f
    μ∂²f[u_comps, u_comps] = ones(u_dim, u_dim)

    # μ∂Δt²f
    μ∂²f[Δt_comp, Δt_comp] = 1.0

    return μ∂²f
end

