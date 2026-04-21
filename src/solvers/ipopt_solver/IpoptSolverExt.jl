module IpoptSolverExt

export Callbacks

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
