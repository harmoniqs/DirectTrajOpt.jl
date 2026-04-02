import MadNLP
import DirectTrajOpt

include("test_utils.jl")

# @assert length([mod for mod = reverse(Base.loaded_modules_order) if Symbol(mod) == :MadNLPSolverExt]) > 1
const MadNLPSolverExt =
    [mod for mod in reverse(Base.loaded_modules_order) if Symbol(mod) == :MadNLPSolverExt][1]
