module IpoptSolverExt

export Callbacks

using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils
using Ipopt
import MadNLP
using MathOptInterface
import MathOptInterface as MOI
import DirectTrajOpt as DTO # Note to self: pick one and stick with it!
using TestItemRunner

using ..Constraints
using ..Integrators
using ..Objectives
using ..Solvers

const AbstractOptimizer = Union{Ipopt.Optimizer, MadNLP.Optimizer}

include("options.jl")
include("constraints.jl")
include("evaluator.jl")
include("solver.jl")
include("callbacks.jl")

end
