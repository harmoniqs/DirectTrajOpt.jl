export GlobalNonlinearConstraint


# ----------------------------------------------------------------------------- #
# GlobalNonlinearConstraint
# ----------------------------------------------------------------------------- #

struct GlobalNonlinearConstraint <: AbstractNonlinearConstraint
    g!::Function
    ∂g!::Function
    ∂gs::SparseMatrixCSC
    μ∂²g!::Function
    μ∂²gs::SparseMatrixCSC
    equality::Bool
    dim::Int

    function GlobalNonlinearConstraint(
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

    function GlobalNonlinearConstraint(
        g::Function,
        global_name::Symbol,
        traj::NamedTrajectory;
        kwargs...
    )
        return GlobalNonlinearConstraint(g, [global_name], traj; kwargs...)
    end
end

function get_full_jacobian(
    NLC::GlobalNonlinearConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    ∂g_full = spzeros(NLC.dim, Z_dim)
    ∂g_full[:, traj.dim * traj.T + 1:Z_dim] = NLC.∂gs
    return ∂g_full
end

function get_full_hessian(
    NLC::GlobalNonlinearConstraint, 
    traj::NamedTrajectory
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    μ∂²g_full = spzeros(Z_dim, Z_dim)
    g_comps = traj.dim * traj.T + 1:Z_dim
    μ∂²g_full[g_comps, g_comps] = NLC.μ∂²gs
    return μ∂²g_full
end

# --------------------------------------------------------------------------- #
# GlobalNonlinearKnotPointConstraint
# --------------------------------------------------------------------------- #

# struct GlobalNonlinearKnotPointConstraint <: AbstractNonlinearConstraint
#     g!::Function
#     ∂g!::Function
#     ∂gs::Vector{SparseMatrixCSC}
#     μ∂²g!::Function
#     μ∂²gs::Vector{SparseMatrixCSC}
#     times::AbstractVector{Int}
#     equality::Bool
#     g_dim::Int
#     dim::Int

#     """
#         TODO: Docstring
#     """
#     function GlobalNonlinearKnotPointConstraint(
#         g::Function,
#         names::AbstractVector{Symbol},
#         traj::NamedTrajectory,
#         params::AbstractVector;
#         global_names::AbstractVector{Symbol}=Symbol[],
#         times::AbstractVector{Int}=1:traj.T,
#         equality::Bool=true,
#         jacobian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
#         hessian_structure::Union{Nothing, SparseMatrixCSC}=nothing,
#     )
#         @assert length(params) == length(times) "params must have the same length as times"

#         Z_dim = traj.dim * traj.T + traj.global_dim
#         x_comps = vcat([traj.components[name] for name in names]...)
#         g_comps = vcat([traj.global_components[name] for name in global_names]...)
#         x_slices = [vcat([slice(t, x_comps, traj.dim), g_comps]...) for t in times]

#         @assert g(traj[times[1]].data[x_comps], params[1]) isa AbstractVector{Float64}
#         g_dim = length(g(traj[times[1]].data[x_comps], params[1]))

#         @views function g!(δ::AbstractVector, Z⃗::AbstractVector)
#             for (i, x_slice) ∈ enumerate(x_slices)
#                 δ[slice(i, g_dim)] = g(Z⃗[x_slice], params[i])
#             end
#         end

#         @views function ∂g!(∂gs::Vector{<:AbstractMatrix}, Z⃗::AbstractVector)
#             for (i, (x_slice, ∂g)) ∈ enumerate(zip(x_slices, ∂gs))
#                 ForwardDiff.jacobian!(
#                     ∂g[:, x_comps], 
#                     x -> g(x, params[i]),
#                     Z⃗[x_slice]
#                 )
#             end
#         end

#         @views function μ∂²g!(
#             μ∂²gs::Vector{<:AbstractMatrix},   
#             Z⃗::AbstractVector, 
#             μ::AbstractVector
#         )
#             for (i, (k, μ∂²g)) ∈ enumerate(zip(times, μ∂²gs))
#                 ForwardDiff.hessian!(
#                     μ∂²g[x_comps, x_comps], 
#                     x -> μ[slice(i, g_dim)]' * g(x, params[i]), 
#                     Z⃗[slice(k, x_comps, traj.dim)]
#                 )
#             end
#         end

#         if isnothing(jacobian_structure)
#             jacobian_structure = spzeros(g_dim, Z_dim) 
#             jacobian_structure[:, x_comps] .= 1.0
#         else
#             @assert size(jacobian_structure) == (g_dim, Z_dim)
#         end

#         ∂gs = [copy(jacobian_structure) for _ ∈ times]

#         if isnothing(hessian_structure)
#             hessian_structure = spzeros(Z_dim, Z_dim) 
#             hessian_structure[x_comps, x_comps] .= 1.0
#         else
#             @assert size(hessian_structure) == (Z_dim, Z_dim)
#         end

#         μ∂²gs = [copy(hessian_structure) for _ ∈ times]

#         return new(
#             g!,
#             ∂g!,
#             ∂gs,
#             μ∂²g!,
#             μ∂²gs,
#             times,
#             equality,
#             g_dim,
#             g_dim * length(times)
#         )
#     end

#     function GlobalNonlinearKnotPointConstraint(
#         g::Function,
#         names::AbstractVector{Symbol},
#         traj::NamedTrajectory;
#         times::AbstractVector{Int}=1:traj.T,
#         kwargs...
#     )
#         params = [nothing for _ in times]
#         g_param = (x, _) -> g(x)
#         return GlobalNonlinearKnotPointConstraint(
#             g_param, 
#             names, 
#             traj, 
#             params; 
#             times=times, 
#             kwargs...
#         )
#     end

#     function GlobalNonlinearKnotPointConstraint(g::Function, name::Symbol, args...; kwargs...)
#         return GlobalNonlinearKnotPointConstraint(g, [name], args...; kwargs...)
#     end

# end

# function get_full_jacobian(
#     NLC::GlobalNonlinearKnotPointConstraint, 
#     traj::NamedTrajectory
# )
#     Z_dim = traj.dim * traj.T + traj.global_dim
#     ∂g_full = spzeros(NLC.dim, Z_dim) 
#     for (i, (k, ∂gₖ)) ∈ enumerate(zip(NLC.times, NLC.∂gs))
#         ∂g_full[slice(i, NLC.g_dim), slice(k, traj.dim)] = ∂gₖ
#     end
#     return ∂g_full
# end

# function get_full_hessian(
#     NLC::GlobalNonlinearKnotPointConstraint, 
#     traj::NamedTrajectory
# )
#     Z_dim = traj.dim * traj.T + traj.global_dim
#     μ∂²g_full = spzeros(Z_dim, Z_dim)
#     for (k, μ∂²gₖ) ∈ zip(NLC.times, NLC.μ∂²gs)
#         μ∂²g_full[slice(k, traj.dim), slice(k, traj.dim)] = μ∂²gₖ
#     end
#     return μ∂²g_full
# end
