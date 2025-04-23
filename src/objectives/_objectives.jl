module Objectives

export Objective
export NullObjective
export KnotPointObjective
export TerminalObjective

using ..Constraints

using TrajectoryIndexingUtils
using NamedTrajectories
using LinearAlgebra
using SparseArrays
using ForwardDiff
using TestItems


# ----------------------------------------------------------------------------- #
#                           Objective                                           #
# ----------------------------------------------------------------------------- #

"""
    Objective

A structure for defining objective functions.
    
Fields:
    `L`: the objective function
    `∇L`: the gradient of the objective function
    `∂²L`: the Hessian of the objective function
    `∂²L_structure`: the structure of the Hessian of the objective function
"""
struct Objective
	L::Function
	∇L::Function
	∂²L::Union{Function, Nothing}
	∂²L_structure::Union{Function, Nothing}
end

function Base.:+(obj1::Objective, obj2::Objective)
	L = Z⃗ -> obj1.L(Z⃗) + obj2.L(Z⃗)
	∇L = Z⃗ -> obj1.∇L(Z⃗) + obj2.∇L(Z⃗)
    ∂²L = Z⃗ -> vcat(obj1.∂²L(Z⃗), obj2.∂²L(Z⃗))
    ∂²L_structure = () -> vcat(obj1.∂²L_structure(), obj2.∂²L_structure())
	return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

function Base.:*(num::Real, obj::Objective)
	L = (Z⃗) -> num * obj.L(Z⃗)
	∇L = (Z⃗) -> num * obj.∇L(Z⃗)
    ∂²L = (Z⃗) -> num * obj.∂²L(Z⃗)
	return Objective(L, ∇L, ∂²L, obj.∂²L_structure)
end

Base.:*(obj::Objective, num::Real) = num * obj

# TODO: Unnecessary?
# Base.:+(obj::Objective, ::Nothing) = obj
# Base.:+(obj::Objective) = obj

# ----------------------------------------------------------------------------- #
# Null objective                                      
# ----------------------------------------------------------------------------- #

function NullObjective(Z::NamedTrajectory)
	L(::AbstractVector{<:Real}) = 0.0
    ∇L(::AbstractVector{R}) where R<:Real = zeros(R, Z.dim * Z.T + Z.global_dim)
    ∂²L_structure() = []
    ∂²L(::AbstractVector{R}) where R<:Real = R[]
	return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

# ----------------------------------------------------------------------------- #
# Knot Point objective
# ----------------------------------------------------------------------------- #

function KnotPointObjective(
    ℓ::Function,
    name::Symbol,
    traj::NamedTrajectory;
    Qs::AbstractVector{Float64}=ones(traj.T),
    times::AbstractVector{Int}=1:traj.T,
)
    @assert length(Qs) == length(times) "Qs must have the same length as times"

    Z_dim = traj.dim * traj.T + traj.global_dim
    x_slices = [slice(t, traj.components[name], traj.dim) for t in times]
    
    function L(Z⃗::AbstractVector{<:Real})
        loss = 0.0
        for (i, x_slice) in enumerate(x_slices)
            x = Z⃗[x_slice]
            loss += Qs[i] * ℓ(x)
        end
        return loss
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real})
        ∇ = zeros(Z_dim)
        for (i, x_slice) in enumerate(x_slices)
            x = Z⃗[x_slice]
            ∇[x_slice] = ForwardDiff.gradient(x -> Qs[i] * ℓ(x), x)
        end
        return ∇
    end

    function ∂²L_structure()
        structure = spzeros(Z_dim, Z_dim)
        for x_slice in x_slices
            structure[x_slice, x_slice] .= 1.0
        end
        structure_pairs = collect(zip(findnz(structure)[1:2]...))
        return structure_pairs
    end

    @views function ∂²L(Z⃗::AbstractVector{<:Real})
        ∂²L_values = zeros(length(∂²L_structure()))
        for (i, x_slice) in enumerate(x_slices)
            ∂²ℓ = ForwardDiff.hessian(x -> Qs[i] * ℓ(x), Z⃗[x_slice])
            ∂²ℓ_length = length(∂²ℓ[:])
            ∂²L_values[(i - 1) * ∂²ℓ_length + 1:i * ∂²ℓ_length] = ∂²ℓ[:]
        end
        return ∂²L_values
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

function TerminalObjective(
    ℓ::Function,
    name::Symbol,
    traj::NamedTrajectory;
    Q::Float64=1.0
)
    return KnotPointObjective(
        ℓ,
        name,
        traj;
        Qs=[Q],
        times=[traj.T]
    )
end

# ----------------------------------------------------------------------------- #
# Additional objectives
# ----------------------------------------------------------------------------- #

include("minimum_time_objective.jl")
include("regularizers.jl")

# =========================================================================== #

# TODO: Testing (see constraints)

end
