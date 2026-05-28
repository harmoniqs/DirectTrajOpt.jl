module Solvers

export AbstractOptimizer
export AbstractSolverOptions, DefaultSolverOptions, _DefaultSolverOptions
export AbstractIntermediateCallback
export _solve
export _solve_with_kwargs
export solve!

import MathOptInterface as MOI
import Ipopt
# import MadNLP
import DirectTrajOpt

import JLD2
import Random
import NamedTrajectories
using TestItemRunner


const AbstractOptimizer = MOI.AbstractOptimizer
abstract type AbstractSolverOptions end

"""
    AbstractIntermediateCallback

Solver-agnostic per-iteration callback for trajectory optimization.

Subtypes implement a callable with signature

    (cb::SubType)(primal::AbstractVector, iter::Integer;
                  obj_value::Float64, inf_pr::Float64, kwargs...) -> Bool

where `primal` is the current full NLP primal vector and `iter` is the
iteration index from the solver's main optimization loop. `obj_value` and
`inf_pr` carry the current objective value and primal infeasibility as
reported by the solver — implementations should accept them as keyword
arguments (and a trailing `kwargs...` to absorb any solver-specific
extras forwarded by future adapters). Return `true` to continue solving,
`false` to stop early (the solver will report a user-requested
termination).

Each solver extension wraps an `AbstractIntermediateCallback` instance in a
solver-specific adapter at solve time, so the same callback object works
with every backend (MadNLP, Ipopt, …).

# Contract

- **`primal` may alias the solver's internal vector.** Copy it (e.g.
  `collect(primal)`) if you need to retain the data past the callback
  invocation — its contents may shift on the next iteration.
- **`iter` is monotonic.** The callback is invoked only from the solver's
  main IPM loop; auxiliary phases (e.g. MadNLP's feasibility restoration
  or robust modes) do not fire it.

# Required MadNLP setup

When using MadNLP, the callback must receive the **full** primal vector
to reconstruct trajectories correctly. MadNLP's default
`fixed_variable_treatment = MakeParameter` eliminates variables with
`lb == ub` from the working primal, so any subtype that maps `primal`
back onto a `NamedTrajectory` needs `fixed_variable_treatment =
MadNLP.RelaxBound`. When an `AbstractIntermediateCallback` is installed
via `MadNLPOptions.intermediate_callback`, DTO sets this automatically
(with an `@info` log) unless the user has provided a value.
"""
abstract type AbstractIntermediateCallback end

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
include("best_pulse.jl")
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
