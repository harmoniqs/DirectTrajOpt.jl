export TimeIntegrator

struct TimeIntegrator <: AbstractIntegrator
    t_name::Symbol

    function TimeIntegrator(
        t::Symbol=:t
    )
        return new(
            t
        )
    end
end

function (D::TimeIntegrator)(
    δₖ::AbstractVector,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int
)
    tₖ = zₖ[D.t_name]
    Δtₖ = zₖ.timestep
    tₖ₊₁ = zₖ₊₁[D.t_name]

    return δₖ .= tₖ₊₁ - tₖ .- Δtₖ
end

function jacobian!(
    ∂D::AbstractMatrix,
    D::TimeIntegrator,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int
)
    # ∂xₖ₊₁D, ∂xₖD, ∂ΔtₖD in jacobian structure
    return nothing
end

function jacobian_structure(D::TimeIntegrator, traj::NamedTrajectory)
    z_dim = traj.dim
    t_comp = traj.components[D.t_name][1]
    Δt_comp = traj.components[traj.timestep][1]

    ∂D = spzeros(1, 2 * z_dim)

    # static components (not updated)

    # ∂xₖ₊₁D
    ∂D[:, z_dim + t_comp] .= 1.0
    # ∂xₖD
    ∂D[:, t_comp] .= -1.0
    # ∂ΔtₖD
    ∂D[:, Δt_comp] .= -1.0

    return ∂D
end

function hessian_of_lagrangian(
    D::TimeIntegrator,
    μₖ::AbstractVector,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int
)
    z_dim = length(zₖ.data)
    return spzeros(2z_dim, 2z_dim) 
end

function hessian_structure(D::TimeIntegrator, traj::NamedTrajectory) 
    z_dim = traj.dim
    return spzeros(2z_dim, 2z_dim) 
end
