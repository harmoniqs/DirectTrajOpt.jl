module Solvers

export AbstractOptimizer
export AbstractSolverOptions, DefaultSolverOptions, _DefaultSolverOptions
export _solve
export solve!

import DirectTrajOpt
import MathOptInterface as MOI
import Ipopt
import MadNLP

using TestItemRunner


const AbstractOptimizer = Union{Ipopt.Optimizer,MadNLP.Optimizer}
abstract type AbstractSolverOptions end

struct DefaultSolverOptions <: AbstractSolverOptions end
const _DefaultSolverOptions::Ref{Type{<:AbstractSolverOptions}} =
    Ref{Type{<:AbstractSolverOptions}}(DefaultSolverOptions)

function _get_DefaultSolverOptions()
    return _DefaultSolverOptions[]
end
function _set_DefaultSolverOptions(optty::Type{<:AbstractSolverOptions})
    _DefaultSolverOptions[] = optty
end

include("constrain.jl")
include("evaluator.jl")
include("solve.jl")

end
