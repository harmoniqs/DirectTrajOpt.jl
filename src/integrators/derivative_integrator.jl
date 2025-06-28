export DerivativeIntegrator

struct DerivativeIntegrator <: AbstractIntegrator
    x_comps::Vector{Int} # e.g. a
    u_comps::Vector{Int} # e.g. ȧ 
    Δt_comp::Int
    z_dim::Int
    x_dim::Int
    u_dim::Int

    function DerivativeIntegrator(
        traj::NamedTrajectory,
        x::Symbol,
        ẋ::Symbol
    )
        @assert traj.dims[x] == traj.dims[ẋ]

        return new(
            traj.components[x],
            traj.components[ẋ],
            traj.components[traj.timestep][1],
            traj.dim,
            traj.dims[x],
            traj.dims[ẋ]
        )
    end
end

function (D::DerivativeIntegrator)(
    δₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector,
    k::Int
)
    aₖ = zₖ[D.x_comps]
    ȧₖ = zₖ[D.u_comps]
    Δtₖ = zₖ[D.Δt_comp]
    aₖ₊₁ = zₖ₊₁[D.x_comps]
    δₖ .= aₖ₊₁ - aₖ - Δtₖ * ȧₖ
    return nothing
end

function jacobian!(
    ∂D::AbstractMatrix,
    D::DerivativeIntegrator,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector,
    k::Int
)
    # ∂ẋₖD 
    Δtₖ = zₖ[D.Δt_comp]
    ∂D[:, D.u_comps] = -Δtₖ * I(D.x_dim)

    # ∂ΔtₖD
    ẋₖ = zₖ[D.u_comps]
    ∂D[:, D.Δt_comp] = -ẋₖ
    return nothing
end

function jacobian_structure(D::DerivativeIntegrator)
    x_dim = D.x_dim
    z_dim = D.z_dim
    x_comps = D.x_comps
    ẋ_comps = D.u_comps

    ∂D = spzeros(x_dim, 2 * z_dim)

    # static components (not updated)

    # ∂xₖ₊₁D
    ∂D[:, z_dim .+ x_comps] = I(x_dim)
    # ∂xₖD
    ∂D[:, x_comps] = -I(x_dim)


    # dynamic (updated)

    # ∂ẋₖD
    ∂D[:, ẋ_comps] = I(x_dim)
    # ∂ΔtₖD
    ∂D[:, D.Δt_comp] = ones(x_dim) 

    return ∂D
end

function hessian_of_lagrangian(
    D::DerivativeIntegrator,
    μₖ::AbstractVector,
    zₖ::AbstractVector,
    zₖ₊₁::AbstractVector,
    k::Int
)
    μ∂²D = spzeros(2D.z_dim, 2D.z_dim)

    # μ∂Δtₖ∂ẋₖD
    μ∂²D[D.u_comps, D.Δt_comp] += -μₖ 
    # μ∂²D[D.Δt_comp, D.u_comps] += -μₖ

    return μ∂²D
end

function hessian_structure(D::DerivativeIntegrator) 
    μ∂²D = spzeros(2D.z_dim, 2D.z_dim) 
    μ∂²D[D.u_comps, D.Δt_comp] .= 1
    # μ∂²D[D.Δt_comp, D.u_comps] .= 1
    return μ∂²D
end

@testitem "testing DerivativeIntegrator" begin
    include("../../test/test_utils.jl")
    traj = named_trajectory_type_1()
    D = DerivativeIntegrator(traj, :a, :da)
    test_integrator(D)
end