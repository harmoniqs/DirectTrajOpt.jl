export TimeIntegrator

struct TimeIntegrator <: AbstractIntegrator
    x_comps::Vector{Int}
    Δt_comp::Int
    z_dim::Int
    x_dim::Int

    function TimeIntegrator(
        traj::NamedTrajectory,
        t::Symbol,
    )
        return new(
            traj.components[t],
            traj.components[traj.timestep][1],
            traj.dim,
            traj.dims[t],
        )
    end
end

function (D::TimeIntegrator)(
    δₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    tₖ = zₖ[D.x_comps]
    Δtₖ = zₖ[D.Δt_comp]
    tₖ₊₁ = zₖ₊₁[D.x_comps]

    return δₖ .= tₖ₊₁ - tₖ .- Δtₖ
end

function jacobian!(
    ∂D::AbstractMatrix,
    D::TimeIntegrator,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    # ∂xₖ₊₁D, ∂xₖD, ∂ΔtₖD in jacobian structure
end

function jacobian_structure(D::TimeIntegrator)
    x_dim = D.x_dim
    z_dim = D.z_dim
    x_comps = D.x_comps

    ∂D = spzeros(x_dim, 2 * z_dim)

    # static components (not updated)

    # ∂xₖ₊₁D
    ∂D[:, z_dim .+ x_comps] = I(x_dim)
    # ∂xₖD
    ∂D[:, x_comps] = -I(x_dim)
    # ∂ΔtₖD
    ∂D[:, D.Δt_comp] = -I(D.x_dim)

    return ∂D
end

function hessian_of_lagrangian(
    D::TimeIntegrator,
    μₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector
)
    return spzeros(2D.z_dim, 2D.z_dim) 
end

function hessian_structure(D::TimeIntegrator) 
    return spzeros(2D.z_dim, 2D.z_dim) 
end
