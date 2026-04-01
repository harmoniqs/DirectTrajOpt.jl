module Solvers


export AbstractOptimizer
export AbstractSolverOptions
export _solve!
export solve!

import MathOptInterface as MOI
import Ipopt
import MadNLP


const AbstractOptimizer = Union{Ipopt.Optimizer,MadNLP.Optimizer}
abstract type AbstractSolverOptions end

struct DefaultSolverOptions <: AbstractSolverOptions end
const _DefaultSolverOptions::Ref{Type{<:AbstractSolverOptions}} = Ref{Type{<:AbstractSolverOptions}}(DefaultSolverOptions)

function constrain! end
include("constrain.jl")

function _solve! end
function solve! end
include("solve.jl")

end
