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

A structure representing an objective function for trajectory optimization.

An objective function consists of the cost function itself along with its first
and second derivatives for gradient-based optimization. The structure supports
automatic differentiation and sparse Hessian representations.

# Fields
- `L::Function`: Objective function that takes trajectory vector Z⃗ and returns a scalar cost
- `∇L::Function`: Gradient function returning ∂L/∂Z⃗
- `∂²L::Union{Function, Nothing}`: Hessian function returning non-zero Hessian entries
- `∂²L_structure::Union{Function, Nothing}`: Function returning sparsity structure of Hessian

# Operators
Objectives support addition and scalar multiplication:
- `obj1 + obj2`: Combine objectives by summing costs and derivatives
- `α * obj`: Scale objective by constant α

# Example
```julia
# Create a regularization objective
obj = QuadraticRegularizer(:u, traj, 1e-2)

# Combine multiple objectives  
total_obj = obj1 + obj2 + 0.1 * obj3
```
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

Base.show(io::IO, ::Objective) = print(io, "Objective(L, ∇L, ∂²L, ∂²L_structure)")

# ----------------------------------------------------------------------------- #
# Null objective                                      
# ----------------------------------------------------------------------------- #

function NullObjective(Z::NamedTrajectory)
	L(::AbstractVector{<:Real}) = 0.0
    ∇L(::AbstractVector{R}) where R<:Real = zeros(R, Z.dim * Z.N + Z.global_dim)
    ∂²L_structure() = []
    ∂²L(::AbstractVector{R}) where R<:Real = R[]
	return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

# ----------------------------------------------------------------------------- #
# Additional objectives
# ----------------------------------------------------------------------------- #

include("knot_point_objectives.jl")
include("global_objectives.jl")
include("minimum_time_objective.jl")
include("regularizers.jl")

end
