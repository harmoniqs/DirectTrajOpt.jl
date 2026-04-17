module MadNLPSolverExt


using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils

import MathOptInterface as MOI
import MadNLP # DO NOT using!

using TestItemRunner


using DirectTrajOpt.Constraints
using DirectTrajOpt.Integrators
using DirectTrajOpt.Objectives
using DirectTrajOpt.Solvers

include("options.jl")
include("solver.jl")

end
