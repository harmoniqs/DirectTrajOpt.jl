export NonlinearKnotPointConstraint

struct NonlinearKnotPointConstraint <: AbstractNonlinearConstraint
    g!::Function
    ∂g!::Function
    ∂gs::Vector{SparseMatrixCSC}
    μ∂²g!::Function
    μ∂²gs::Vector{SparseMatrixCSC}
    equality::Bool
    times::AbstractVector{Int}
    g_dim::Int
    dim::Int

    function NonlinearKnotPointConstraint(
        g::Function,
        name::Symbol,
        traj::NamedTrajectory;
        equality::Bool=true,
        times::AbstractVector{Int}=1:traj.T,
        jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
        hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
    )
        @assert g(traj[1][name]) isa AbstractVector{Float64}

        g_dim = length(g(traj[1][name]))
        z_dim = traj.dim
        x_comps = traj.components[name]

        @views function g!(δ::AbstractVector, Z⃗::AbstractVector)
            for (i, k) ∈ enumerate(times)
                δ[slice(i, g_dim)] = g(Z⃗[slice(k, x_comps, z_dim)])
            end
        end

        @views function ∂g!(∂gs::Vector{<:AbstractMatrix}, Z⃗::AbstractVector)
            for (k, ∂g) ∈ zip(times, ∂gs)
                ForwardDiff.jacobian!(
                    ∂g[:, x_comps], 
                    g, 
                    Z⃗[slice(k, x_comps, z_dim)]
                )
            end
        end

        @views function μ∂²g!(
            μ∂²gs::Vector{<:AbstractMatrix},   
            Z⃗::AbstractVector, 
            μ::AbstractVector
        )
            for (i, (k, μ∂²g)) ∈ enumerate(zip(times, μ∂²gs))
                ForwardDiff.hessian!(
                    μ∂²g[x_comps, x_comps], 
                    Z -> μ[slice(i, g_dim)]' * g(Z), 
                    Z⃗[slice(k, x_comps, z_dim)]
                )
            end
        end

        if isnothing(jacobian_structure)
            jacobian_structure = spzeros(g_dim, z_dim) 
            jacobian_structure[:, x_comps] .= 1.0
        else
            @assert size(jacobian_structure) == (g_dim, z_dim)
        end

        ∂gs = [copy(jacobian_structure) for _ ∈ times]

        if isnothing(hessian_structure)
            hessian_structure = spzeros(z_dim, z_dim) 
            hessian_structure[x_comps, x_comps] .= 1.0
        else
            @assert size(hessian_structure) == (z_dim, z_dim)
        end

        μ∂²gs = [copy(hessian_structure) for _ ∈ times]

        return new(
            g!,
            ∂g!,
            ∂gs,
            μ∂²g!,
            μ∂²gs,
            equality,
            times,
            g_dim,
            g_dim * length(times)
        )
    end
end

function get_full_jacobian(
    NLC::NonlinearKnotPointConstraint, 
    traj::NamedTrajectory
)
    ∂g_full = spzeros(NLC.dim, traj.dim * traj.T) 
    for (i, (k, ∂gₖ)) ∈ enumerate(zip(NLC.times, NLC.∂gs))
        ∂g_full[slice(i, NLC.g_dim), slice(k, traj.dim)] = ∂gₖ
    end
    return ∂g_full
end

function get_full_hessian(
    NLC::NonlinearKnotPointConstraint, 
    traj::NamedTrajectory
)
    μ∂²g_full = spzeros(traj.dim * traj.T, traj.dim * traj.T)
    for (k, μ∂²gₖ) ∈ zip(NLC.times, NLC.μ∂²gs)
        μ∂²g_full[slice(k, traj.dim), slice(k, traj.dim)] = μ∂²gₖ
    end
    return μ∂²g_full
end

@testitem "testing NonlinearConstraint" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(a) = [norm(a) - 1.0]

    times = 1:traj.T

    NLC = NonlinearKnotPointConstraint(g, :u, traj; times=times)

    ĝ(Z⃗) = vcat([g(Z⃗[slice(k, traj.components[:u], traj.dim)]) for k ∈ times]...)

    δ = zeros(length(times))

    NLC.g!(δ, traj.datavec)

    @test δ ≈ ĝ(traj.datavec)
    
    NLC.∂g!(NLC.∂gs, traj.datavec)

    ∂g_full = Constraints.get_full_jacobian(NLC, traj)

    ∂g_autodiff = ForwardDiff.jacobian(ĝ, traj.datavec)

    @test ∂g_full ≈ ∂g_autodiff

    μ = randn(length(times))

    NLC.μ∂²g!(NLC.μ∂²gs, traj.datavec, μ)

    hessian_autodiff = ForwardDiff.hessian(Z -> μ'ĝ(Z), traj.datavec)

    μ∂²g_full = Constraints.get_full_hessian(NLC, traj) 

    @test μ∂²g_full ≈ hessian_autodiff
end

