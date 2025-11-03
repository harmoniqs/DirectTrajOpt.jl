export DerivativeIntegrator

struct DerivativeIntegrator <: AbstractIntegrator
    x_name::Symbol
    ẋ_name::Symbol

    function DerivativeIntegrator(
        traj::NamedTrajectory,
        x::Symbol,
        ẋ::Symbol
    )
        @assert traj.dims[x] == traj.dims[ẋ]

        return new(
            x,
            ẋ
        )
    end
end

function (D::DerivativeIntegrator)(
    δₖ::AbstractVector,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int
)
    aₖ = zₖ[D.x_name]
    ȧₖ = zₖ[D.ẋ_name]
    Δtₖ = zₖ.timestep
    aₖ₊₁ = zₖ₊₁[D.x_name]
    δₖ .= aₖ₊₁ - aₖ - Δtₖ * ȧₖ
    return nothing
end

function jacobian!(
    ∂D::AbstractMatrix,
    D::DerivativeIntegrator,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int
)
    # ∂ẋₖD 
    Δtₖ = zₖ.timestep
    ẋ_comps = zₖ.components[D.ẋ_name]
    x_dim = length(zₖ[D.x_name])
    ∂D[:, ẋ_comps] = -Δtₖ * I(x_dim)

    # ∂ΔtₖD
    ẋₖ = zₖ[D.ẋ_name]
    Δt_comp = zₖ.components[zₖ.names[findfirst(==(:Δt), zₖ.names)]][1]
    ∂D[:, Δt_comp] = -ẋₖ
    return nothing
end

function jacobian_structure(D::DerivativeIntegrator, traj::NamedTrajectory)
    x_dim = traj.dims[D.x_name]
    z_dim = traj.dim
    x_comps = traj.components[D.x_name]
    ẋ_comps = traj.components[D.ẋ_name]
    Δt_comp = traj.components[traj.timestep][1]

    ∂D = spzeros(x_dim, 2 * z_dim)

    # static components (not updated)

    # ∂xₖ₊₁D
    ∂D[:, z_dim .+ x_comps] = I(x_dim)
    # ∂xₖD
    ∂D[:, x_comps] = -I(x_dim)


    # dynamic (updated)

    # ∂ẋₖD
    ∂D[:, ẋ_comps] = I(x_dim)
    # ∂ΔtₖD
    ∂D[:, Δt_comp] = ones(x_dim) 

    return ∂D
end

function hessian_of_lagrangian(
    D::DerivativeIntegrator,
    μₖ::AbstractVector,
    zₖ::KnotPoint,
    zₖ₊₁::KnotPoint,
    k::Int
)
    z_dim = length(zₖ.data)
    ẋ_comps = zₖ.components[D.ẋ_name]
    Δt_comp = zₖ.components[zₖ.names[findfirst(==(:Δt), zₖ.names)]][1]
    
    μ∂²D = spzeros(2z_dim, 2z_dim)

    # μ∂Δtₖ∂ẋₖD
    μ∂²D[ẋ_comps, Δt_comp] += -μₖ 
    # μ∂²D[Δt_comp, ẋ_comps] += -μₖ

    return μ∂²D
end

function hessian_structure(D::DerivativeIntegrator, traj::NamedTrajectory) 
    z_dim = traj.dim
    ẋ_comps = traj.components[D.ẋ_name]
    Δt_comp = traj.components[traj.timestep][1]
    
    μ∂²D = spzeros(2z_dim, 2z_dim) 
    μ∂²D[ẋ_comps, Δt_comp] .= 1
    # μ∂²D[Δt_comp, ẋ_comps] .= 1
    return μ∂²D
end

@testitem "testing DerivativeIntegrator" begin
    include("../../test/test_utils.jl")
    traj = named_trajectory_type_1()
    D = DerivativeIntegrator(traj, :a, :da)
    test_integrator(D, traj)
end
