module IpoptSolverExt

export Callbacks # temporarily closing down callbacks; not yet sure whether keeping them in here or generalizing them for MOI more generally would be best

import MathOptInterface as MOI
using MathOptInterface
using Ipopt

import DirectTrajOpt as DTO # Note to self: pick one and stick with it!
using TestItemRunner

using DirectTrajOpt
using NamedTrajectories
using TrajectoryIndexingUtils

using ..Constraints
using ..Integrators
using ..Objectives
using ..Solvers

include("options.jl")
include("solver.jl")
include("callbacks.jl")

end
