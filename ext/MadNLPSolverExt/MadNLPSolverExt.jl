module MadNLPSolverExt

import MathOptInterface as MOI
import MadNLP # DO NOT using!

const MadNLPMOI = Base.get_extension(MadNLP, :MadNLPMOI)

using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils

using TestItemRunner


using DirectTrajOpt.Constraints
using DirectTrajOpt.Integrators
using DirectTrajOpt.Objectives
using DirectTrajOpt.Solvers


# function __init__()
#     @assert MadNLP isa Module
    
# end


include("options.jl")
# include("solver.jl")

end
