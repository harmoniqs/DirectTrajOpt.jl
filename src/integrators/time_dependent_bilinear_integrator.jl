export TimeDependentBilinearIntegrator

using OrdinaryDiffEqTsit5
using ForwardDiff

# -------------------------------------------------------------------------------- #
# Time Dependent Bilinear Integrator
# -------------------------------------------------------------------------------- #

struct TimeDependentBilinearIntegrator{F} <: AbstractBilinearIntegrator
    G::F
    probs::Vector{ODEProblem}
    x_comps::Vector{Int}
    u_comps::Vector{Int}
    t_comp::Int
    Δt_comp::Int
    z_dim::Int
    x_dim::Int
    u_dim::Int
    linear_spline::Bool

    function TimeDependentBilinearIntegrator(
        G::F,
        traj::NamedTrajectory,
        x::Symbol,
        u::Symbol,
        t::Symbol;
        linear_spline::Bool = false
    ) where F <: Function

        @assert traj.N > 1 "Trajectory must have at least two timesteps."
        
        function f!(dx, x_, p, τ)
            t_, Δt, u_ = p[1], p[2], p[3:end]
            if linear_spline
                uₖ = u_[1:length(u_)÷2]
                uₖ₊₁ = u_[length(u_)÷2+1:end]
                u_fn = s -> uₖ .+ s * (uₖ₊₁ .- uₖ)
            else
                u_fn = s -> u_
            end
            mul!(dx, G(u_fn(τ), t_ + τ * Δt), x_ * Δt)
            return nothing
        end

        x_comp = traj.components[x]
        u_comp = traj.components[u]
        u_dim = traj.dims[u]

        x₀ = zeros(length(x_comp))

        if linear_spline
            u₀ = zeros(2u_dim)
        else
            u₀ = zeros(u_dim)
        end

        t₀ = 0.0
        Δt₀ = 1.0
        probs = [
            ODEProblem(f!, x₀, (0.0, 1.0), [t₀; Δt₀; u₀...])
            for _ in 1:traj.N - 1
        ]

        return new{F}(
            G,
            probs,
            x_comp,
            u_comp,
            traj.components[t][1],
            traj.components[traj.timestep][1],
            traj.dim,
            traj.dims[x],
            traj.dims[u],
            linear_spline
        )
    end
end

@views function (B::TimeDependentBilinearIntegrator)(
    δₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector,
    k::Int;
    # atol=1e-6,
    # rtol=1e-6,
    kwargs...
)
    xₖ = zₖ[B.x_comps]
    xₖ₊₁ = zₖ₊₁[B.x_comps]
    uₖ = zₖ[B.u_comps]
    uₖ₊₁ = zₖ₊₁[B.u_comps]
    tₖ = zₖ[B.t_comp]
    Δtₖ = zₖ[B.Δt_comp]

    if B.linear_spline
        pₖ = [tₖ; Δtₖ; uₖ; uₖ₊₁]
    else
        pₖ = [tₖ; Δtₖ; uₖ]
    end

    probₖ = remake(B.probs[k], u0 = xₖ, p = pₖ)
    solₖ = solve(probₖ, Tsit5();  kwargs...)
    δₖ[:] = xₖ₊₁ - solₖ[:, end]
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

    # ∂uₖ₊₁f
    ∂f[:, z_dim .+ u_comps] = ones(x_dim, u_dim)

    # ∂tₖf
    ∂f[:, t_comp] = ones(x_dim)

    # ∂Δtₖf
    ∂f[:, Δt_comp] = ones(x_dim)

    return ∂f
end

function hessian_structure(B::TimeDependentBilinearIntegrator)

    if B.linear_spline
        p_comps = [B.t_comp; B.Δt_comp; B.u_comps; B.z_dim .+ B.u_comps]
    else
        p_comps = [B.t_comp; B.Δt_comp; B.u_comps]
    end

    μ∂²f = spzeros(2 * B.z_dim, 2 * B.z_dim)

    μ∂²f[B.x_comps, p_comps] .= 1.0

    μ∂²f[p_comps, p_comps] .= 1.0

    return sparse(UpperTriangular(μ∂²f))
end

# ============================================================================ #

@testitem "testing zoh TimeDependentBilinearIntegrator" begin
    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_time=true)

    # zero order hold
    B = TimeDependentBilinearIntegrator((a, t) -> G(a), traj, :x, :u, :t)

    test_integrator(
        B, test_equality=false, atol=0.0, rtol=5e-2, reltol=1e-6, abstol=1e-6
    )
end