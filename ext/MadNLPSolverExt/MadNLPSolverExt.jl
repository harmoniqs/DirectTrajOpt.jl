module MadNLPSolverExt

import MathOptInterface as MOI
import MadNLP # DO NOT using!

# Several potential ways to ensure MadNLP.Optimizer() accessible within `solver.jl`:
#   - (currently working) use AbstractOptimizer (alias of MOI.AbstractOptimizer) in signatures and MadNLP.Optimizer in code; not ideal given the parent package's prolific use of @reexport
#   - (currently broken) bring in MadNLPMOI (via Base.get_extension) so that it exists at compile time (in signatures and code)
#   - (not yet tested) use the module __init__() routine

using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils

using TestItemRunner


using DirectTrajOpt.Constraints
using DirectTrajOpt.Integrators
using DirectTrajOpt.Objectives
using DirectTrajOpt.Solvers


# function __init__()
#     return nothing
# end


include("options.jl")
include("solver.jl")

end
