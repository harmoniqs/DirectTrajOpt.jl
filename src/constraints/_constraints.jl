module Constraints

export AbstractConstraint

export AbstractLinearConstraint
export AbstractNonlinearConstraint

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

include("linear_constraints.jl")
include("nonlinear_constraints.jl")
include("nonlinear_global_constraints.jl")

end
