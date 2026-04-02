module MadNLPSolverExt


using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils

import MathOptInterface as MOI
import MadNLP

using TestItemRunner


using ..Constraints
using ..Integrators
using ..Objectives
using ..Solvers

include("options.jl")
include("solver.jl")

end
