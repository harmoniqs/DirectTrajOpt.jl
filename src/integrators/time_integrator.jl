export TimeIntegrator

struct TimeIntegrator <: AbstractIntegrator
    t_name::Symbol

    function TimeIntegrator(
        traj::NamedTrajectory,
        t::Symbol,
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
    x_dim = traj.dims[D.t_name]
    z_dim = traj.dim
    x_comps = traj.components[D.t_name]
    Δt_comp = traj.components[traj.timestep][1]

    ∂D = spzeros(x_dim, 2 * z_dim)

    # static components (not updated)

    # ∂xₖ₊₁D
    ∂D[:, z_dim .+ x_comps] = I(x_dim)
    # ∂xₖD
    ∂D[:, x_comps] = -I(x_dim)
    # ∂ΔtₖD
    ∂D[:, Δt_comp] = -I(x_dim)

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
