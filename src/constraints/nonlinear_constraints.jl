export NonlinearKnotPointConstraint


struct NonlinearKnotPointConstraint{F1, F2, F3} <: AbstractNonlinearConstraint
    g!::F1
    ∂g!::F2
    ∂gs::Vector{SparseMatrixCSC}
    μ∂²g!::F3
    μ∂²gs::Vector{SparseMatrixCSC}
    times::AbstractVector{Int}
    equality::Bool
    g_dim::Int
    dim::Int

    """
        NonlinearKnotPointConstraint(
            g::Function,
            names::AbstractVector{Symbol},
            traj::NamedTrajectory,
            params::AbstractVector;
            kwargs...
        )
        NonlinearKnotPointConstraint(
            g::Function,
            names::AbstractVector{Symbol},
            traj::NamedTrajectory;
            kwargs...
        )
        NonlinearKnotPointConstraint(
            g::Function,
            name::Symbol,
            args...;
            kwargs...
        )

    Create a NonlinearKnotPointConstraint object that represents a nonlinear constraint `g(x,p)`
    on a trajectory variable `x` with parameters `p`. If the parameters argument is omitted, 
    `g(x)` is assumed to be a function of `x` only.

    # Arguments
    - `g::Function`: Function that defines the constraint, g(x, p) or g(x).
    - `name::Symbol`: Name of the variable to be constrained.
    - `traj::NamedTrajectory`: The trajectory on which the constraint is defined.
    - `params::AbstractVector`: Parameters `p` for the constraint function `g`, for each time.

    # Keyword Arguments
    - `times::AbstractVector{Int}=1:traj.T`: Time indices at which the constraint is enforced.
    - `equality::Bool=true`: If `true`, the constraint is `g(x) = 0`. Otherwise, the constraint is `g(x) ≤ 0`.
    - `jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing`: Structure of the Jacobian matrix of the constraint.
    - `hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing`: Structure of the Hessian matrix of the constraint.
    """
    function NonlinearKnotPointConstraint(
        g::Function,
        names::AbstractVector{Symbol},
        traj::NamedTrajectory,
        params::AbstractVector;
        times::AbstractVector{Int}=1:traj.T,
        equality::Bool=true,
        jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
        hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
    )
        @assert length(params) == length(times) "params must have the same length as times"

        z_dim = traj.dim
        x_comps = vcat([traj.components[name] for name in names]...)
        x_slices = [slice(t, x_comps, traj.dim) for t in times]

        @assert g(traj[times[1]].data[x_comps], params[1]) isa AbstractVector{Float64}
        g_dim = length(g(traj[times[1]].data[x_comps], params[1]))

        @views function g!(δ::AbstractVector, Z⃗::AbstractVector)
            for (i, x_slice) ∈ enumerate(x_slices)
                δ[slice(i, g_dim)] = g(Z⃗[x_slice], params[i])
            end
        end

        @views function ∂g!(∂gs::Vector{<:AbstractMatrix}, Z⃗::AbstractVector)
            for (i, (x_slice, ∂g)) ∈ enumerate(zip(x_slices, ∂gs))
                ForwardDiff.jacobian!(
                    ∂g[:, x_comps], 
                    x -> g(x, params[i]),
                    Z⃗[x_slice]
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
                    x -> μ[slice(i, g_dim)]' * g(x, params[i]), 
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

        return new{typeof(g!), typeof(∂g!), typeof(μ∂²g!)}(
            g!,
            ∂g!,
            ∂gs,
            μ∂²g!,
            μ∂²gs,
            times,
            equality,
            g_dim,
            g_dim * length(times)
        )
    end

    function NonlinearKnotPointConstraint(
        g::Function,
        names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        times::AbstractVector{Int}=1:traj.T,
        kwargs...
    )
        params = [nothing for _ in times]
        g_param = (x, _) -> g(x)
        return NonlinearKnotPointConstraint(
            g_param, 
            names, 
            traj, 
            params; 
            times=times, 
            kwargs...
        )
    end

    function NonlinearKnotPointConstraint(g::Function, name::Symbol, args...; kwargs...)
        return NonlinearKnotPointConstraint(g, [name], args...; kwargs...)
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

# ============================================================================= #

@testitem "testing NonlinearConstraint" begin

    using TrajectoryIndexingUtils
    
    include("../../test/test_utils.jl")

    _, traj = bilinear_dynamics_and_trajectory()

    g(a) = [norm(a) - 1.0]

    times = 1:traj.T

    NLC = NonlinearKnotPointConstraint(g, :u, traj; times=times, equality=false)

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

