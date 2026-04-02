module IpoptSolverExt

# export Callbacks # temporarily closing down callbacks; not yet sure whether keeping them in here or generalizing them for MOI more generally would be best

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

include("options.jl")
include("evaluator.jl")
include("solver.jl")
# include("callbacks.jl")

end
