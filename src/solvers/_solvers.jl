module Solvers


export AbstractOptimizer
export AbstractSolverOptions
export _solve!
export solve!

import MathOptOptimizer as MOI
import Ipopt
import MadNLP


const AbstractOptimizer = Union{Ipopt.Optimizer,MadNLP.Optimizer}
abstract type AbstractSolverOptions end

function constrain! end
include("constrain.jl")

function _solve! end
function solve! end
include("solve.jl")

end
