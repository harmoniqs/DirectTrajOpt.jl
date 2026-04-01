module Solvers


export AbstractOptimizer
export AbstractSolverOptions, DefaultSolverOptions, _DefaultSolverOptions
export _solve
export solve!

import MathOptInterface as MOI
import Ipopt
import MadNLP


const AbstractOptimizer = Union{Ipopt.Optimizer,MadNLP.Optimizer}
abstract type AbstractSolverOptions end

struct DefaultSolverOptions <: AbstractSolverOptions end
const _DefaultSolverOptions::Ref{Type{<:AbstractSolverOptions}} = Ref{Type{<:AbstractSolverOptions}}(DefaultSolverOptions)

function _get_DefaultSolverOptions()
    return _DefaultSolverOptions[]
end
function _set_DefaultSolverOptions(optty::Type{<:AbstractSolverOptions})
    _DefaultSolverOptions[] = optty
end


function constrain! end
include("constrain.jl")

function _solve end
function solve! end
include("solve.jl")

end
