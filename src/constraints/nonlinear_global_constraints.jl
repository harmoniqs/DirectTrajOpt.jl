export NonlinearGlobalConstraint
export NonlinearGlobalKnotPointConstraint


# ----------------------------------------------------------------------------- #
# NonlinearGlobalConstraint
# ----------------------------------------------------------------------------- #

struct NonlinearGlobalConstraint <: AbstractNonlinearConstraint
    g!::Function
    ∂g!::Function
    ∂gs::SparseMatrixCSC
    μ∂²g!::Function
    μ∂²gs::SparseMatrixCSC
    equality::Bool
    dim::Int

    function NonlinearGlobalConstraint(
        g::Function,
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        equality::Bool=true,
    )
        global_comps = vcat([traj.global_components[name] for name in global_names]...)
        local_comps = global_comps .- traj.dim * traj.T

        g_eval = g(vec(traj)[global_comps])
        @assert g_eval isa AbstractVector{Float64}
        g_dim = length(g_eval)
        
        @views function g!(δ::AbstractVector, Z⃗::AbstractVector)
            δ[:] = g(Z⃗[global_comps])
            return nothing
        end

        @views function ∂g!(∂g::AbstractMatrix, Z⃗::AbstractVector)
            ForwardDiff.jacobian!(
                ∂g, 
                x -> g(x),
                Z⃗[global_comps]
            )
        end

        # global subspace
        jacobian_structure = spzeros(g_dim, traj.global_dim) 
        jacobian_structure[:, local_comps] .= 1.0

        @views function μ∂²g!(
            μ∂²g::AbstractMatrix,   
            Z⃗::AbstractVector, 
            μ::AbstractVector
        )
            ForwardDiff.hessian!(
                μ∂²g[local_comps, local_comps], 
                x -> μ'g(x), 
                Z⃗[global_comps]
            )
        end

        # global subspace
        hessian_structure = spzeros(traj.global_dim, traj.global_dim) 
        hessian_structure[local_comps, local_comps] .= 1.0

        return new(
            g!,
            ∂g!,
            jacobian_structure,
            μ∂²g!,
            hessian_structure,
            equality,
            g_dim,
        )
    end

    function NonlinearGlobalConstraint(
        g::Function,
        global_name::Symbol,
        traj::NamedTrajectory;
        kwargs...
    )
        return NonlinearGlobalConstraint(g, [global_name], traj; kwargs...)
    end
end

function get_full_jacobian(
    NLC::NonlinearGlobalConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    ∂g_full = spzeros(NLC.dim, Z_dim)
    ∂g_full[:, traj.dim * traj.T + 1:Z_dim] = NLC.∂gs
    return ∂g_full
end

function get_full_hessian(
    NLC::NonlinearGlobalConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    g_comps = traj.dim * traj.T + 1:Z_dim
    μ∂²g_full[g_comps, g_comps] = NLC.μ∂²gs
    return μ∂²g_full
end

# --------------------------------------------------------------------------- #
# NonlinearGlobalKnotPointConstraint
# --------------------------------------------------------------------------- #

struct NonlinearGlobalKnotPointConstraint <: AbstractNonlinearConstraint
    g!::Function
    ∂g!::Function
    ∂gs::Vector{SparseMatrixCSC}
    μ∂²g!::Function
    μ∂²gs::Vector{SparseMatrixCSC}
    times::AbstractVector{Int}
    equality::Bool
    g_dim::Int
    dim::Int

    """
        TODO: Docstring
    """
    function NonlinearGlobalKnotPointConstraint(
        g::Function,
        names::AbstractVector{Symbol},
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory,
        params::AbstractVector;
        times::AbstractVector{Int}=1:traj.T,
        equality::Bool=true,
        jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
        hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
    )
        @assert length(params) == length(times) "params must have the same length as times"

        x_comps = vcat([traj.components[name] for name in names]...)
        global_comps = vcat([traj.global_components[name] for name in global_names]...)
        local_comps = global_comps .- traj.dim * traj.T

        # (Rebased) global data is appended to the knot point
        xg_comps = vcat([x_comps, traj.dim .+ local_comps]...)
        z_dim = traj.dim + traj.global_dim

        # Each slice indexes into Z⃗
        xg_slices = [vcat([slice(t, x_comps, traj.dim), global_comps]...) for t in times]

        Z⃗ = vec(traj)
        @assert g(Z⃗[xg_slices[1]], params[1]) isa AbstractVector{Float64}
        g_dim = length(g(Z⃗[xg_slices[1]], params[1]))

        @views function g!(δ::AbstractVector, Z⃗::AbstractVector)
            for (i, xg_slice) ∈ enumerate(xg_slices)
                δ[slice(i, g_dim)] = g(Z⃗[xg_slice], params[i])
            end
        end

        @views function ∂g!(∂gs::Vector{<:AbstractMatrix}, Z⃗::AbstractVector)
            for (i, (xg_slice, ∂g)) ∈ enumerate(zip(xg_slices, ∂gs))
                # Overlapping
                ∂g[:, xg_comps] .+= ForwardDiff.jacobian(
                    xg -> g(xg, params[i]), 
                    Z⃗[xg_slice]
                )
            end
        end

        @views function μ∂²g!(
            μ∂²gs::Vector{<:AbstractMatrix},   
            Z⃗::AbstractVector, 
            μ::AbstractVector
        )
            for (i, (xg_slice, μ∂²g)) ∈ enumerate(zip(xg_slices, μ∂²gs))
                # Overlapping
                μ∂²g[xg_comps, xg_comps] .+= ForwardDiff.hessian(
                    xg -> μ[slice(i, g_dim)]' * g(xg, params[i]), 
                    Z⃗[xg_slice]
                )
            end
        end

        if isnothing(jacobian_structure)
            jacobian_structure = spzeros(g_dim, z_dim) 
            jacobian_structure[:, xg_comps] .= 1.0
        else
            @assert size(jacobian_structure) == (g_dim, z_dim)
        end

        ∂gs = [copy(jacobian_structure) for _ ∈ times]

        if isnothing(hessian_structure)
            hessian_structure = spzeros(z_dim, z_dim) 
            hessian_structure[xg_comps, xg_comps] .= 1.0
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
            times,
            equality,
            g_dim,
            g_dim * length(times)
        )
    end

    function NonlinearGlobalKnotPointConstraint(
        g::Function,
        names::AbstractVector{Symbol},
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        times::AbstractVector{Int}=1:traj.T,
        kwargs...
    )
        params = [nothing for _ in times]
        g_param = (x, _) -> g(x)
        return NonlinearGlobalKnotPointConstraint(
            g_param, 
            names,
            global_names,
            traj, 
            params; 
            times=times, 
            kwargs...
        )
    end

    function NonlinearGlobalKnotPointConstraint(g::Function, name::Symbol, args...; kwargs...)
        return NonlinearGlobalKnotPointConstraint(g, [name], args...; kwargs...)
    end

end

function get_full_jacobian(
    NLC::NonlinearGlobalKnotPointConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    global_slice = traj.dim * traj.T .+ (1:traj.global_dim)
    ∂g_full = spzeros(NLC.dim, Z_dim) 
    for (i, (k, ∂gₖ)) ∈ enumerate(zip(NLC.times, NLC.∂gs))
        # Overlapping
        zg_slice = vcat([slice(k, traj.dim), global_slice]...)
        ∂g_full[slice(i, NLC.g_dim), zg_slice] .+= ∂gₖ
    end
    return ∂g_full
end

function get_full_hessian(
    NLC::NonlinearGlobalKnotPointConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    global_slice = traj.dim * traj.T .+ (1:traj.global_dim)
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    for (k, μ∂²gₖ) ∈ zip(NLC.times, NLC.μ∂²gs)
        # Overlapping
        zg_slice = vcat([slice(k, traj.dim), global_slice]...)
        μ∂²g_full[zg_slice, zg_slice] .+= μ∂²gₖ
    end
    return μ∂²g_full
end
