module Constraints

export AbstractConstraint

export AbstractLinearConstraint
export AbstractNonlinearConstraint

export constraint_value
export jacobian_structure
export jacobian!
export hessian_structure
export hessian_of_lagrangian

using NamedTrajectories
using TrajectoryIndexingUtils
using ForwardDiff
using SparseArrays
using TestItemRunner

# ----------------------------------------------------------------------------- #
#                     Abstract Constraints                                      #
# ----------------------------------------------------------------------------- #

abstract type AbstractConstraint end
abstract type AbstractLinearConstraint <: AbstractConstraint end
abstract type AbstractNonlinearConstraint <: AbstractConstraint end

# ----------------------------------------------------------------------------- #
#                     Abstract Constraint Interface                             #
# ----------------------------------------------------------------------------- #

"""
    constraint_value(constraint, Z⃗::AbstractVector)

Evaluate the constraint at the given trajectory vector.
"""
function constraint_value end

"""
    jacobian_structure(constraint, traj::NamedTrajectory)

Return the sparsity structure of the constraint Jacobian.
"""
function jacobian_structure end

"""
    jacobian!(∂g, constraint, Z⃗::AbstractVector)

Compute the Jacobian of the constraint in-place.
"""
function jacobian! end

"""
    hessian_structure(constraint, traj::NamedTrajectory)

Return the sparsity structure of the constraint Hessian.
"""
function hessian_structure end

"""
    hessian_of_lagrangian(constraint, μ::AbstractVector, Z⃗::AbstractVector)

Compute the Hessian of the Lagrangian (μ'g) for the constraint.
"""
function hessian_of_lagrangian end

# Linear constraints
include("linear/equality_constraint.jl")
include("linear/all_equal_constraint.jl")
include("linear/bounds_constraint.jl")
include("linear/total_constraint.jl")
include("linear/symmetry_constraint.jl")

# Nonlinear constraints
include("nonlinear/knot_point_constraint.jl")
include("nonlinear/global_constraint.jl")
include("nonlinear/global_knot_point_constraint.jl")

end
