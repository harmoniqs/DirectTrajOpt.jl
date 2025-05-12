export TimeDependentBilinearIntegrator

using DifferentialEquations
using SciMLBase
using SciMLSensitivity


struct TimeDependentBilinearIntegrator <: AbstractBilinearIntegrator
    G::Function
    prob::ODEProblem
    x_comps::AbstractVector{Int}
    u_comps::AbstractVector{Int}
    t_comp::Int
    Δt_comp::Int
    z_dim::Int
    x_dim::Int
    u_dim::Int

    function TimeDependentBilinearIntegrator(
        G::Function,
        traj::NamedTrajectory,
        x::Symbol,
        u::Symbol,
        t::Symbol
    )

        function f!(dx, x, p, τ)
            t, Δt, u = p[1], p[2], p[3:end]
            dx[:] = G(u, t + τ * Δt) * x
        end

        x₀ = zeros(traj.components[x])
        u₀ = zeros(traj.components[u])
        t₀ = 0.0
        Δt₀ = 1.0
        prob = ODEProblem(f!, x₀, (0.0, 1.0), [t₀; Δt₀; u₀...])
        return new(
            G,
            prob,
            traj.components[x],
            traj.components[u],
            traj.components[t][1],
            traj.components[traj.timestep][1],
            traj.dim,
            traj.dims[x],
            traj.dims[u],
        )
    end
end

@views function (B::TimeDependentBilinearIntegrator)(
    δₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    xₖ₊₁ = zₖ₊₁[B.x_comps]
    xₖ = zₖ[B.x_comps]
    uₖ = zₖ[B.u_comps]
    tₖ = zₖ[B.t_comp]
    Δtₖ = zₖ[B.Δt_comp]

    _prob = remake(B.prob, u0 = xₖ, p = [tₖ, Δtₖ, uₖ...])
    sol = solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
    δₖ[:] = xₖ₊₁ - sol[:,end]
end

@views function jacobian!(
    ∂f::AbstractMatrix,
    B!::TimeDependentBilinearIntegrator,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    # function f(zₖ)
    #     xₖ = zₖ[B!.x_comps]
    #     uₖ = zₖ[B!.u_comps]
    #     tₖ = zₖ[B!.t_comp]
    #     Δtₖ = zₖ[B!.Δt_comp]
    #     _prob = remake(B!.prob, u0 = xₖ, p = [tₖ; Δtₖ; uₖ])
    #     solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
    # end
    #∂f[:, 1:B!.z_dim] = ForwardDiff.jacobian!(∂f, f, zeros(B!.x_dim), [zₖ])
    ForwardDiff.jacobian!(
        ∂f,
        (δ, zz) -> B!(δ, zz[1:B!.z_dim], zz[B!.z_dim+1:end]),
        zeros(B!.x_dim),
        [zₖ; zₖ₊₁]
    )
    #∂f[:, B!.z_dim .+ B!.x_comps] = I(B!.x_dim)
end

function jacobian_structure(B::TimeDependentBilinearIntegrator)

    z_dim = B.z_dim
    x_dim = B.x_dim
    u_dim = B.u_dim

    x_comps = B.x_comps
    u_comps = B.u_comps
    t_comp = B.t_comp
    Δt_comp = B.Δt_comp

    ∂f = spzeros(x_dim, 2 * z_dim)

    # ∂xₖ₊₁f
    ∂f[:, z_dim .+ x_comps] = I(x_dim)

    # ∂xₖf
    ∂f[:, x_comps] = ones(x_dim, x_dim)

    # ∂uₖf
    ∂f[:, u_comps] = ones(x_dim, u_dim)

    # ∂tₖf
    ∂f[:, t_comp] = ones(x_dim)

    # ∂Δtₖf
    ∂f[:, Δt_comp] = ones(x_dim)

    return ∂f
end


# @views function hessian_of_lagrangian(
#     B!::TimeDependentBilinearIntegrator,
#     μₖ::AbstractVector,
#     zₖ::AbstractVector,
#     zₖ₊₁::AbstractVector
# )

#     function f(zₖ)
#         xₖ = zₖ[B.x_comps]
#         uₖ = zₖ[B.u_comps]
#         tₖ = zₖ[B.t_comp]
#         Δtₖ = zₖ[B.Δt_comp]
#         _prob = remake(prob, u0 = xₖ, p = [tₖ; Δtₖ; uₖ])
#         solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
#     end
#     μ∂²f[:B.z_dim,:B.z_dim] = ForwardDiff.hessian!(f, zₖ)
    
# end

function hessian_structure(B::TimeDependentBilinearIntegrator)

    x_comps = B.x_comps
    u_comps = B.u_comps
    t_comp = B.t_comp
    Δt_comp = B.Δt_comp

    x_dim = B.x_dim
    u_dim = B.u_dim

    μ∂²f = spzeros(2 * B.z_dim, 2 * B.z_dim)

    # μ∂ₓₖ∂ᵤf & μ∂ᵤ∂ₓₖf
    μ∂²f[x_comps, u_comps] = ones(x_dim, u_dim)

    # μ∂ₓₖ∂ₜf & μ∂ₜ∂ₓₖf
    μ∂²f[x_comps, t_comp] = ones(x_dim)

    # μ∂ₓₖ∂Δtₖf & μ∂Δtₖ∂ₓₖf
    μ∂²f[x_comps, Δt_comp] = ones(x_dim)

    # μ∂u∂tf & μ∂t∂uf
    μ∂²f[u_comps,t_comp] = ones(u_dim)

    # μ∂u∂Δtₖf & μ∂Δtₖ∂uf
    μ∂²f[u_comps, Δt_comp] = ones(u_dim)

    # μ∂t∂Δtₖf & μ∂Δtₖ∂tf
    μ∂²f[t_comp, Δt_comp] = 1.0

    # μ∂ᵤ²f
    μ∂²f[u_comps, u_comps] = ones(u_dim, u_dim)

    # μ∂ₜ²f
    μ∂²f[t_comp, t_comp] = 1.0

    # μ∂Δt²f
    μ∂²f[Δt_comp, Δt_comp] = 1.0

    return μ∂²f
end