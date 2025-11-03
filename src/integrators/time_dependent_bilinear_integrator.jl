export TimeDependentBilinearIntegrator

using OrdinaryDiffEqTsit5
using ForwardDiff

# -------------------------------------------------------------------------------- #
# Time Dependent Bilinear Integrator
# -------------------------------------------------------------------------------- #

struct TimeDependentBilinearIntegrator{F} <: AbstractBilinearIntegrator
    G::F
    probs::Vector{ODEProblem}
    x_name::Symbol
    u_name::Symbol
    t_name::Symbol
    linear_spline::Bool

    function TimeDependentBilinearIntegrator(
        G::F,
        x::Symbol,
        u::Symbol,
        t::Symbol,
        x_dim::Int,
        u_dim::Int,
        N::Int;
        linear_spline::Bool = false
    ) where F <: Function

        @assert N > 1 "Trajectory must have at least two timesteps."
        
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

        x₀ = zeros(x_dim)

        if linear_spline
            u₀ = zeros(2u_dim)
        else
            u₀ = zeros(u_dim)
        end

        t₀ = 0.0
        Δt₀ = 1.0
        probs = [
            ODEProblem(f!, x₀, (0.0, 1.0), [t₀; Δt₀; u₀...])
            for _ in 1:N - 1
        ]

        return new{F}(
            G,
            probs,
            x,
            u,
            t,
            linear_spline
        )
    end
end

@views function (B::TimeDependentBilinearIntegrator)(
    δₖ::AbstractVector,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int;
    # atol=1e-6,
    # rtol=1e-6,
    kwargs...
)
    xₖ = zₖ[B.x_name]
    xₖ₊₁ = zₖ₊₁[B.x_name]
    uₖ = zₖ[B.u_name]
    uₖ₊₁ = zₖ₊₁[B.u_name]
    tₖ = zₖ[B.t_name]
    Δtₖ = zₖ.timestep

    if B.linear_spline
        pₖ = [tₖ; Δtₖ; uₖ; uₖ₊₁]
    else
        pₖ = [tₖ; Δtₖ; uₖ]
    end

    probₖ = remake(B.probs[k], u0 = xₖ, p = pₖ)
    solₖ = solve(probₖ, Tsit5();  kwargs...)
    δₖ[:] = xₖ₊₁ - solₖ[:, end]
end

function jacobian_structure(B::TimeDependentBilinearIntegrator, traj::NamedTrajectory)

    z_dim = traj.dim
    x_dim = traj.dims[B.x_name]
    u_dim = traj.dims[B.u_name]

    x_comps = traj.components[B.x_name]
    u_comps = traj.components[B.u_name]
    t_comp = traj.components[B.t_name][1]
    Δt_comp = traj.components[traj.timestep][1]

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

function hessian_structure(B::TimeDependentBilinearIntegrator, traj::NamedTrajectory)

    z_dim = traj.dim
    t_comp = traj.components[B.t_name][1]
    Δt_comp = traj.components[traj.timestep][1]
    u_comps = traj.components[B.u_name]
    x_comps = traj.components[B.x_name]

    if B.linear_spline
        p_comps = [t_comp; Δt_comp; u_comps; z_dim .+ u_comps]
    else
        p_comps = [t_comp; Δt_comp; u_comps]
    end

    μ∂²f = spzeros(2 * z_dim, 2 * z_dim)

    μ∂²f[x_comps, p_comps] .= 1.0

    μ∂²f[p_comps, p_comps] .= 1.0

    return sparse(UpperTriangular(μ∂²f))
end

# ============================================================================ #

@testitem "testing TimeDependentBilinearIntegrator" begin
    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_time=true)

    # zero order hold
    B = TimeDependentBilinearIntegrator(
        (a, t) -> G(a), 
        :x, :u, :t, 
        traj.dims[:x], traj.dims[:u], traj.N
    )

    test_integrator(
        B, traj, test_equality=false, rtol=1e-4, atol=1e-4
    )
end