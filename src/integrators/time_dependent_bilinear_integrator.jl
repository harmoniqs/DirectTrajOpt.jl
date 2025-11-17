export TimeDependentBilinearIntegrator

using OrdinaryDiffEqTsit5
using TrajectoryIndexingUtils

# -------------------------------------------------------------------------------- #
# Time-Dependent Bilinear Integrator
# -------------------------------------------------------------------------------- #

struct TimeDependentBilinearIntegrator{F} <: AbstractBilinearIntegrator
    f::F
    x_name::Symbol
    u_name::Symbol
    t_name::Symbol
    spline_order::Int
    x_dim::Int
    u_dim::Int
    dim::Int

    function TimeDependentBilinearIntegrator(
        G::F,
        x::Symbol,
        u::Symbol,
        t::Symbol,
        traj::NamedTrajectory;
        spline_order::Int=1,
        solve_kwargs = (;)
    ) where F <: Function

        N = traj.N
        @assert N > 1 "Trajectory must have at least two timesteps."

        x_dim = traj.dims[x]
        u_dim = traj.dims[u]

        # Build template ODE problems once per timestep


        if spline_order == 0
            u_fn = (τ, pₖ) -> pₖ
        elseif spline_order == 1
            u_fn = (τ, pₖ) -> begin
                uₖ = pₖ[1:u_dim] 
                uₖ₊₁ = pₖ[u_dim+1:2u_dim] 
                return uₖ + τ * (uₖ₊₁ - uₖ)
            end
        else
            error("Unsupported spline order: $spline_order")
        end

        function f!(dx, x_, p, τ)
            pₖ, Δtₖ, tₖ = p[1:end-2], p[end-1], p[end]
            mul!(dx, G(u_fn(τ, pₖ), tₖ + τ * Δtₖ), x_ * Δtₖ)
            return nothing
        end

        u_template = if spline_order == 0 
            zeros(u_dim)
        elseif spline_order == 1
            zeros(2u_dim)
        else
            error("Unsupported spline order: $spline_order")
        end 

        p_template = vcat(u_template, 1.0, 0.0) # [controls..., Δt, t]

        prob_template = ODEProblem(f!, zeros(x_dim), (0.0, 1.0), p_template)

        dim = x_dim * (N - 1)

        solve_kwargs_nt = (; solve_kwargs...)

        f = (xₖ₊₁, xₖ, pₖ, Δtₖ, tₖ) -> begin
            prob = remake(prob_template, u0 = xₖ, p = [pₖ; Δtₖ; tₖ])
            sol = solve(prob, Tsit5(); solve_kwargs_nt...)
            return xₖ₊₁ - sol[:, end]
        end

        return new{typeof(f)}(
            f,
            x,
            u,
            t,
            spline_order,
            x_dim,
            u_dim,
            dim
        )
    end
end

# -------------------------------------------------------------------------------- #
# Methods
# -------------------------------------------------------------------------------- #

function evaluate!(
    δ::AbstractVector,
    B::TimeDependentBilinearIntegrator,
    traj::NamedTrajectory;
    kwargs...
)
    for k = 1:traj.N-1
        xₖ = traj[k][B.x_name]
        xₖ₊₁ = traj[k+1][B.x_name]
        uₖ = traj[k][B.u_name]
        tₖ = traj[k][B.t_name][1]
        Δtₖ = traj[k].timestep

        if B.spline_order == 0
            pₖ = uₖ
        elseif B.spline_order == 1
            uₖ₊₁ = traj[k+1][B.u_name]
            pₖ = [uₖ; uₖ₊₁]
        else
            error("Unsupported spline order: $(B.spline_order)")
        end

        δ[slice(k, B.x_dim)] = B.f(xₖ₊₁, xₖ, pₖ, tₖ, Δtₖ)
    end
    return nothing
end

# Jacobian methods

@views function eval_jacobian(
    B::TimeDependentBilinearIntegrator,
    traj::NamedTrajectory
)
    ∂B = spzeros(B.dim, traj.dim * traj.N + traj.global_dim)
    for k = 1:traj.N-1
        ForwardDiff.jacobian!(
            ∂B[slice(k, B.x_dim), slice(k, 1:2traj.dim, traj.dim)],
            zz -> begin 
                zₖ = zz[1:traj.dim]
                zₖ₊₁ = zz[traj.dim+1:end]
                xₖ = zₖ[traj.components[B.x_name]]
                uₖ = zₖ[traj.components[B.u_name]]
                tₖ = zₖ[traj.components[B.t_name]][1]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]
                xₖ₊₁ = zₖ₊₁[traj.components[B.x_name]]
                
                if B.spline_order == 0
                    pₖ = uₖ
                elseif B.spline_order == 1
                    uₖ₊₁ = zₖ₊₁[traj.components[B.u_name]]
                    pₖ = [uₖ; uₖ₊₁]
                else
                    error("Unsupported spline order: $(B.spline_order)")
                end

                return B.f(xₖ₊₁, xₖ, pₖ, tₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )
    end
    return ∂B 
end

# Hessian methods

function eval_hessian_of_lagrangian(
    B::TimeDependentBilinearIntegrator,
    traj::NamedTrajectory,
    μ::AbstractVector
)
    μ∂²B = spzeros(
        traj.dim * traj.N + traj.global_dim,
        traj.dim * traj.N + traj.global_dim,
    )

    for k = 1:traj.N-1
        μₖ = μ[slice(k, B.x_dim)]
       
        μ∂²Bₖ = ForwardDiff.hessian(
            zz -> begin
                zₖ = zz[1:traj.dim]
                zₖ₊₁ = zz[traj.dim+1:end]
                xₖ = zₖ[traj.components[B.x_name]]
                uₖ = zₖ[traj.components[B.u_name]]
                tₖ = zₖ[traj.components[B.t_name]][1]
                Δtₖ = zₖ[traj.components[traj.timestep]][1]
                xₖ₊₁ = zₖ₊₁[traj.components[B.x_name]]

                if B.spline_order == 0
                    pₖ = uₖ
                elseif B.spline_order == 1
                    uₖ₊₁ = zₖ₊₁[traj.components[B.u_name]]
                    pₖ = [uₖ; uₖ₊₁]
                else
                    error("Unsupported spline order: $(B.spline_order)")
                end

                return μₖ'B.f(xₖ₊₁, xₖ, pₖ, tₖ, Δtₖ)
            end,
            [traj[k].data; traj[k+1].data],
        )

        μ∂²B[slice(k, 1:2traj.dim, traj.dim), slice(k, 1:2traj.dim, traj.dim)] .+= μ∂²Bₖ
    end
    return μ∂²B 
end

# ============================================================================ #

@testitem "testing TimeDependentBilinearIntegrator" begin
    include("../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_time=true)

    # zero order hold
    B = TimeDependentBilinearIntegrator(
        (a, t) -> G(a), 
        :x, :u, :t, 
        traj
    )

    test_integrator(
        B, traj, test_equality=false, atol=1e-3
    )
end