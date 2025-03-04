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


# include("quantum_objective.jl")
# include("unitary_infidelity_objective.jl")
include("regularizers.jl")
include("minimum_time_objective.jl")
include("terminal_loss.jl")
# include("unitary_robustness_objective.jl")
# include("density_operator_objectives.jl")

# TODO:
# - [ ] Do not reference the Z object in the objective (components only / remove "name")

"""
    sparse_to_moi(A::SparseMatrixCSC)

Converts a sparse matrix to tuple of vector of nonzero indices and vector of nonzero values
"""

# ----------------------------------------------------------------------------- #
#                           Objective                                           #
# ----------------------------------------------------------------------------- #

"""
    Objective

A structure for defining objective functions.

The `terms` field contains all the arguments needed to construct the objective function.

Fields:
    `L`: the objective function
    `∇L`: the gradient of the objective function
    `∂²L`: the Hessian of the objective function
    `∂²L_structure`: the structure of the Hessian of the objective function
    `terms`: a vector of dictionaries containing the terms of the objective function
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

# Base.:+(obj::Objective, ::Nothing) = obj
# Base.:+(obj::Objective) = obj

# function Objective(terms::AbstractVector{<:Dict})
#     return +(Objective.(terms)...)
# end

# function Base.:*(num::Real, obj::Objective)
# 	L = (Z⃗, Z) -> num * obj.L(Z⃗, Z)
# 	∇L = (Z⃗, Z) -> num * obj.∇L(Z⃗, Z)
#     if isnothing(obj.∂²L)
#         ∂²L = nothing
#         ∂²L_structure = nothing
#     else
#         ∂²L = (Z⃗, Z) -> num * obj.∂²L(Z⃗, Z)
#         ∂²L_structure = obj.∂²L_structure
#     end
# 	return Objective(L, ∇L, ∂²L, ∂²L_structure, obj.terms)
# end

# Base.:*(obj::Objective, num::Real) = num * obj

# function Objective(term::Dict)
#     return eval(term[:type])(; delete!(term, :type)...)
# end

# ----------------------------------------------------------------------------- #
#                           Null objective                                      #
# ----------------------------------------------------------------------------- #

function NullObjective()
	L(Z⃗::AbstractVector{R}) where R<:Real = 0.0
    ∇L(Z⃗::AbstractVector{R}) where R<:Real = zeros(R, Z.dim * Z.T + Z.global_dim)
    ∂²L_structure(Z::NamedTrajectory) = []
    function ∂²L(Z⃗::AbstractVector{<:Real}; return_moi_vals=true)
        n = Z.dim * Z.T + Z.global_dim
        return return_moi_vals ? [] : spzeros(n, n)
    end
	return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

end
