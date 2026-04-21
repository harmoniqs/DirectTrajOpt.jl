module Solvers

export AbstractOptimizer
export AbstractSolverOptions, DefaultSolverOptions, _DefaultSolverOptions
export _solve
export _solve_with_kwargs
export solve!

import MathOptInterface as MOI
import Ipopt
# import MadNLP
import DirectTrajOpt

using TestItemRunner


const AbstractOptimizer = MOI.AbstractOptimizer
abstract type AbstractSolverOptions end

struct DefaultSolverOptions <: AbstractSolverOptions end
const _DefaultSolverOptions::Ref{Type{<:AbstractSolverOptions}} =
    Ref{Type{<:AbstractSolverOptions}}(DefaultSolverOptions)

function _get_DefaultSolverOptions()
    return _DefaultSolverOptions[]
end
function _set_DefaultSolverOptions(optty::Type{<:AbstractSolverOptions})
    _DefaultSolverOptions[] = optty
end

include("constrain.jl")
include("evaluator.jl")
include("solve.jl")


# Coverage targets: src/solvers/_solvers.jl (50% → ~100%)

@testitem "DefaultSolverOptions get/set" setup=[DTOTestHelpers] begin
    original = Solvers._get_DefaultSolverOptions()
    @test original == IpoptSolverExt.IpoptOptions

    Solvers._set_DefaultSolverOptions(DirectTrajOpt.MadNLPOptions)
    @test Solvers._get_DefaultSolverOptions() == DirectTrajOpt.MadNLPOptions

    Solvers._set_DefaultSolverOptions(Solvers.DefaultSolverOptions)
    @test Solvers._get_DefaultSolverOptions() == Solvers.DefaultSolverOptions

    Solvers._set_DefaultSolverOptions(original)
    @test Solvers._get_DefaultSolverOptions() == IpoptSolverExt.IpoptOptions
end

@testitem "AbstractOptimizer alias" setup=[DTOTestHelpers] begin
    @test Solvers.AbstractOptimizer === MOI.AbstractOptimizer
end

@testitem "AbstractSolverOptions hierarchy" setup=[DTOTestHelpers] begin
    @test IpoptSolverExt.IpoptOptions <: Solvers.AbstractSolverOptions
    @test DirectTrajOpt.MadNLPOptions <: Solvers.AbstractSolverOptions
    @test Solvers.DefaultSolverOptions <: Solvers.AbstractSolverOptions
end

end
