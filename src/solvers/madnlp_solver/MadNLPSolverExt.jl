module MadNLPSolverExt


using DirectTrajOpt
# using NamedTrajectories
# using TrajectoryIndexingUtils

import MathOptInterface as MOI
import MadNLP

using TestItemRunner


using ..Solvers

include("options.jl")
include("solver.jl")

end
