module MadNLPSolverExt

import MathOptInterface as MOI
import MadNLP # DO NOT using!

using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils

using TestItemRunner


using DirectTrajOpt.Constraints
using DirectTrajOpt.Integrators
using DirectTrajOpt.Objectives
using DirectTrajOpt.Solvers


include("options.jl")
include("solver.jl")
include("utils.jl")

end
