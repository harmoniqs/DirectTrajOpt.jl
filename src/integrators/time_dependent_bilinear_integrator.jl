export TimeDependentBilinearIntegrator

using OrdinaryDiffEq
using ForwardDiff

# -------------------------------------------------------------------------------- #
# Time Dependent Bilinear Integrator
# -------------------------------------------------------------------------------- #

struct TimeDependentBilinearIntegrator{F} <: AbstractBilinearIntegrator
    G::F
    prob::ODEProblem
    x_comps::Vector{Int}
    u_comps::Vector{Int}
    t_comp::Int
    Δt_comp::Int
    z_dim::Int
    x_dim::Int
    u_dim::Int

    function TimeDependentBilinearIntegrator(
        G::F,
        traj::NamedTrajectory,
        x::Symbol,
        u::Symbol,
        t::Symbol
    ) where F <: Function

        function f!(dx, x_, p, τ)
            t_, Δt, u_ = p[1], p[2], p[3:end]
            dx[:] = G(u_, t_ + τ * Δt) * x_ * Δt
            return nothing
        end

        x_comp = traj.components[x]
        u_comp = traj.components[u]

        x₀ = zeros(length(x_comp))
        u₀ = zeros(length(u_comp))
        t₀ = 0.0
        Δt₀ = 1.0
        prob = ODEProblem(f!, x₀, (0.0, 1.0), [t₀; Δt₀; u₀...])

        return new{F}(
            G,
            prob,
            x_comp,
            u_comp,
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
    zₖ₊₁::AbstractVector;
    algorithm::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=Tsit5(),
    # atol=1e-6,
    # rtol=1e-6,
    kwargs...
)
    xₖ₊₁ = zₖ₊₁[B.x_comps]
    xₖ = zₖ[B.x_comps]
    uₖ = zₖ[B.u_comps]
    tₖ = zₖ[B.t_comp]
    Δtₖ = zₖ[B.Δt_comp]

    _prob = remake(B.prob, u0 = xₖ, p = [tₖ, Δtₖ, uₖ...])
    sol = solve(_prob, algorithm;  kwargs...)
    δₖ[:] = xₖ₊₁ - sol[:, end]
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

@testitem "testing zoh TimeDependentBilinearIntegrator" begin
    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(time=true)

    # zero order hold
    B = TimeDependentBilinearIntegrator((a, t) -> G(a), traj, :x, :u, :t)

    test_integrator(
        B, test_equality=false, atol=0.0, rtol=5e-2, reltol=1e-6, abstol=1e-6
    )
end