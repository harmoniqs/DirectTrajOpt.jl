module Objectives

export Objective
export NullObjective

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
struct Objective{F1, F2, F3, F4}
	L::F1
	∇L::F2
	∂²L::Union{F3, Nothing}
	∂²L_structure::Union{F4, Nothing}
end

function Base.:+(obj1::Objective, obj2::Objective)
	L = Z⃗ -> obj1.L(Z⃗) + obj2.L(Z⃗)
	∇L = Z⃗ -> obj1.∇L(Z⃗) + obj2.∇L(Z⃗)
    ∂²L = Z⃗ -> vcat(obj1.∂²L(Z⃗), obj2.∂²L(Z⃗))
    ∂²L_structure = () -> vcat(obj1.∂²L_structure(), obj2.∂²L_structure())
	return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

# TODO: Unnecessary base case?
# Base.:+(obj::Objective, ::Nothing) = obj
# Base.:+(obj::Objective) = obj

function Base.:+(num::Real, obj::Objective)
	L = (Z⃗) -> num + obj.L(Z⃗)
	return Objective(L, obj.∇L, obj.∂²L, obj.∂²L_structure)
end

Base.:+(obj::Objective, num::Real) = num + obj

function Base.:*(num::Real, obj::Objective)
	L = (Z⃗) -> num * obj.L(Z⃗)
	∇L = (Z⃗) -> num * obj.∇L(Z⃗)
    ∂²L = (Z⃗) -> num * obj.∂²L(Z⃗)
	return Objective(L, ∇L, ∂²L, obj.∂²L_structure)
end

Base.:*(obj::Objective, num::Real) = num * obj


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
# Additional objectives
# ----------------------------------------------------------------------------- #

include("knot_point_objectives.jl")
include("minimum_time_objective.jl")
include("regularizers.jl")

end
