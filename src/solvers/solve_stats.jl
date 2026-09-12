"""
    SolveStats

What the solver actually did, as data — returned by `solve!` (and
`DirectTrajOpt._solve`) instead of `nothing`. Both backends (Ipopt, MadNLP)
populate it from their optimizers after `MOI.optimize!`.

Fields:
- `status::MOI.TerminationStatusCode` — the MOI termination code
- `raw_status::String` — backend-specific status string (e.g. Ipopt's
  "Maximum Number of Iterations Exceeded.")
- `objective_value::Float64` — the final NLP objective (includes the barrier
  terms for interior-point methods, not the pure trajectory objective)
- `iterations::Int` — IPM iterations taken (Ipopt `BarrierIterations`;
  MadNLP `solve_iterations`)
- `solve_time_s::Float64` — wall time of `MOI.optimize!` (excludes setup)
- `solver::Symbol` — `:ipopt` or `:madnlp`
"""
Base.@kwdef struct SolveStats
    status::MOI.TerminationStatusCode
    raw_status::String
    objective_value::Float64
    iterations::Int
    solve_time_s::Float64
    solver::Symbol
end

function _solve_stats(optimizer, variables, solver::Symbol, t0::Float64)
    status = MOI.get(optimizer, MOI.TerminationStatus())
    raw = try
        MOI.get(optimizer, MOI.RawStatusString())
    catch
        string(status)
    end
    obj = try
        Float64(MOI.get(optimizer, MOI.ObjectiveValue()))
    catch
        NaN
    end
    iters = try
        Int(MOI.get(optimizer, MOI.BarrierIterations()))
    catch
        -1
    end
    return SolveStats(;
        status = status,
        raw_status = raw,
        objective_value = obj,
        iterations = iters,
        solve_time_s = time() - t0,
        solver = solver,
    )
end

# ----------------------------------------------------------------------------
# Status vocabulary (C2 — spec-20260830-madnlp-first-flip, amendment A8):
# a closed Symbol classification of SolveStats.status, defined ONCE for both
# arms. MadNLP's cnt.status reaches SolveStats.status through MadNLPMOI's
# _STATUS_CODES translation, so the vocabulary's domain is the
# MOI.TerminationStatusCode set.
# ----------------------------------------------------------------------------

"""
    Solvers.solve_status_symbol(status) -> Symbol

Classify a solve's `MOI.TerminationStatusCode` (the `status` field of
[`SolveStats`](@ref), populated identically on the Ipopt and MadNLP arms)
into a closed Symbol vocabulary, exhaustive over the statuses the stack's
NLP backends can terminate with:

| symbol         | class     | meaning | MOI codes |
|:---------------|:----------|:--------|:----------|
| `:solved`      | success   | converged (incl. "acceptable level") | `OPTIMAL`, `LOCALLY_SOLVED`, `ALMOST_OPTIMAL`, `ALMOST_LOCALLY_SOLVED` |
| `:limit`       | failure   | a resource limit bound the solve | `ITERATION_LIMIT`, `TIME_LIMIT`, `OBJECTIVE_LIMIT`, `OTHER_LIMIT` |
| `:infeasible`  | failure   | converged to (local) infeasibility / diverging iterates | `INFEASIBLE`, `LOCALLY_INFEASIBLE`, `DUAL_INFEASIBLE`, `ALMOST_INFEASIBLE`, `ALMOST_DUAL_INFEASIBLE`, `INFEASIBLE_OR_UNBOUNDED` |
| `:numerical`   | failure   | numerical trouble (restoration failure, tiny steps) | `NUMERICAL_ERROR`, `SLOW_PROGRESS`, `NORM_LIMIT` |
| `:invalid`     | failure   | model/option invalid (e.g. MadNLP's `NOT_ENOUGH_DEGREES_OF_FREEDOM`) | `INVALID_MODEL`, `INVALID_OPTION` |
| `:interrupted` | failure   | user requested stop | `INTERRUPTED` |
| `:error`       | failure   | solver-internal error | `OTHER_ERROR` |
| `:not_called`  | neutral   | `MOI.optimize!` never ran | `OPTIMIZE_NOT_CALLED` |
| `:unknown`     | —         | status outside the vocabulary (loud `@warn`) | — |

Every code in MadNLP's termination-status set (`MadNLP.Status` →
`MadNLPMOI._STATUS_CODES`) and Ipopt's is mapped. The MIP/branch-and-bound-only
codes `NODE_LIMIT`, `SOLUTION_LIMIT`, and `MEMORY_LIMIT` are unreachable from
an NLP backend and are **declared unmapped**: a backend one day terminating
with one degrades loudly to `:unknown` rather than silently reusing a
misleading class. Anything that is not an `MOI.TerminationStatusCode` at all
takes the same loud `:unknown` path.
"""
const _STATUS_VOCABULARY = (
    :solved,
    :limit,
    :infeasible,
    :numerical,
    :invalid,
    :interrupted,
    :error,
    :not_called,
    :unknown,
)

const _STATUS_SYMBOLS = Dict{MOI.TerminationStatusCode,Symbol}(
    # success class
    MOI.OPTIMAL => :solved,
    MOI.LOCALLY_SOLVED => :solved,
    MOI.ALMOST_OPTIMAL => :solved,
    MOI.ALMOST_LOCALLY_SOLVED => :solved,
    # limits
    MOI.ITERATION_LIMIT => :limit,
    MOI.TIME_LIMIT => :limit,
    MOI.OBJECTIVE_LIMIT => :limit,
    MOI.OTHER_LIMIT => :limit,
    # infeasibility / divergence
    MOI.INFEASIBLE => :infeasible,
    MOI.LOCALLY_INFEASIBLE => :infeasible,
    MOI.DUAL_INFEASIBLE => :infeasible,
    MOI.ALMOST_INFEASIBLE => :infeasible,
    MOI.ALMOST_DUAL_INFEASIBLE => :infeasible,
    MOI.INFEASIBLE_OR_UNBOUNDED => :infeasible,
    # numerical trouble
    MOI.NUMERICAL_ERROR => :numerical,
    MOI.SLOW_PROGRESS => :numerical,
    MOI.NORM_LIMIT => :numerical,
    # invalid model / options
    MOI.INVALID_MODEL => :invalid,
    MOI.INVALID_OPTION => :invalid,
    # user stop
    MOI.INTERRUPTED => :interrupted,
    # solver-internal error
    MOI.OTHER_ERROR => :error,
    # neutral
    MOI.OPTIMIZE_NOT_CALLED => :not_called,
)

# Declared-unmapped: MIP / branch-and-bound-only codes no NLP backend on this
# stack (Ipopt, MadNLP) can terminate with. Kept OUT of _STATUS_SYMBOLS so an
# unexpected code degrades loudly (see solve_status_symbol) instead of silently
# reusing a misleading class.
const _STATUS_UNMAPPED = [MOI.NODE_LIMIT, MOI.SOLUTION_LIMIT, MOI.MEMORY_LIMIT]

function _warn_unmapped_status(status)
    @warn "Unmapped solve status $status — add it to Solvers._STATUS_SYMBOLS (or _STATUS_UNMAPPED if " *
          "no NLP backend can emit it); reporting :unknown"
    return nothing
end

function solve_status_symbol(status::MOI.TerminationStatusCode)::Symbol
    symbol = get(_STATUS_SYMBOLS, status, nothing)
    if symbol === nothing
        _warn_unmapped_status(status)
        return :unknown
    end
    return symbol
end

function solve_status_symbol(status)::Symbol
    _warn_unmapped_status(status)
    return :unknown
end

@testitem "solve! returns SolveStats (Ipopt)" setup=[DTOTestHelpers] begin
    using DirectTrajOpt.Solvers: SolveStats
    using Ipopt

    prob, _ = make_standard_prob()
    stats = solve!(prob; options = IpoptOptions(max_iter = 5), verbose = false)

    @test stats isa SolveStats
    @test stats.solver === :ipopt
    @test stats.iterations > 0
    @test isfinite(stats.objective_value)
    @test stats.solve_time_s >= 0
    @test !isempty(stats.raw_status)
    @test stats.status isa MOI.TerminationStatusCode
end

@testitem "coverage: _solve_stats fallbacks when the optimizer lacks an API" setup =
    [DTOTestHelpers] begin
    using DirectTrajOpt.Solvers: _solve_stats

    # A minimal stand-in optimizer: only TerminationStatus is supported, so
    # the raw-status, objective-value, and barrier-iteration getters all
    # take their fallbacks (string(status), NaN, -1).
    struct _NoStatsOptimizer end
    MOI.get(::_NoStatsOptimizer, ::MOI.TerminationStatus) = MOI.LOCALLY_SOLVED
    MOI.get(::_NoStatsOptimizer, ::MOI.RawStatusString) = error("unsupported")
    MOI.get(::_NoStatsOptimizer, ::MOI.ObjectiveValue) = error("unsupported")
    MOI.get(::_NoStatsOptimizer, ::MOI.BarrierIterations) = error("unsupported")

    stats = _solve_stats(_NoStatsOptimizer(), nothing, :mock, time())
    @test stats.status === MOI.LOCALLY_SOLVED
    @test stats.raw_status == string(MOI.LOCALLY_SOLVED)
    @test isnan(stats.objective_value)
    @test stats.iterations == -1
    @test stats.solver === :mock
    @test stats.solve_time_s >= 0
end

# ----------------------------------------------------------------------------
# Status vocabulary (C2 — spec-20260830-madnlp-first-flip A8): every
# TerminationStatusCode either maps to a Symbol or is EXPLICITLY declared
# unmapped (loud :unknown + warn at runtime).
# ----------------------------------------------------------------------------

@testitem "status vocabulary: exhaustive partition of MOI.TerminationStatusCode" begin
    using DirectTrajOpt.Solvers: _STATUS_SYMBOLS, _STATUS_UNMAPPED, _STATUS_VOCABULARY
    import MathOptInterface as MOI

    codes = instances(MOI.TerminationStatusCode)
    # Exhaustive: every code is either mapped or explicitly unmapped, never both.
    @test Set(keys(_STATUS_SYMBOLS)) ∪ Set(_STATUS_UNMAPPED) == Set(codes)
    @test isdisjoint(keys(_STATUS_SYMBOLS), _STATUS_UNMAPPED)
    @test length(_STATUS_SYMBOLS) + length(_STATUS_UNMAPPED) == length(codes)
    # The mapped symbols all come from the documented closed vocabulary.
    @test Set(values(_STATUS_SYMBOLS)) ⊆ Set(_STATUS_VOCABULARY)
    # The unmapped set is exactly the MIP/branch-and-bound-only codes no NLP
    # backend on this stack (Ipopt, MadNLP) can terminate with.
    @test Set(_STATUS_UNMAPPED) ==
          Set([MOI.NODE_LIMIT, MOI.SOLUTION_LIMIT, MOI.MEMORY_LIMIT])
end

@testitem "status vocabulary: symbol spot checks across the success and failure classes" begin
    using DirectTrajOpt.Solvers: solve_status_symbol
    import MathOptInterface as MOI

    # success class
    @test solve_status_symbol(MOI.OPTIMAL) === :solved
    @test solve_status_symbol(MOI.LOCALLY_SOLVED) === :solved
    @test solve_status_symbol(MOI.ALMOST_OPTIMAL) === :solved
    @test solve_status_symbol(MOI.ALMOST_LOCALLY_SOLVED) === :solved
    # limit class
    @test solve_status_symbol(MOI.ITERATION_LIMIT) === :limit
    @test solve_status_symbol(MOI.TIME_LIMIT) === :limit
    # infeasible class
    @test solve_status_symbol(MOI.INFEASIBLE) === :infeasible
    @test solve_status_symbol(MOI.LOCALLY_INFEASIBLE) === :infeasible
    @test solve_status_symbol(MOI.INFEASIBLE_OR_UNBOUNDED) === :infeasible
    # numerical class
    @test solve_status_symbol(MOI.NUMERICAL_ERROR) === :numerical
    @test solve_status_symbol(MOI.SLOW_PROGRESS) === :numerical
    # invalid class
    @test solve_status_symbol(MOI.INVALID_MODEL) === :invalid
    @test solve_status_symbol(MOI.INVALID_OPTION) === :invalid
    # stop / error / neutral
    @test solve_status_symbol(MOI.INTERRUPTED) === :interrupted
    @test solve_status_symbol(MOI.OTHER_ERROR) === :error
    @test solve_status_symbol(MOI.OPTIMIZE_NOT_CALLED) === :not_called
end

@testitem "status vocabulary: unknown status degrades to :unknown with a loud log" begin
    using DirectTrajOpt.Solvers: solve_status_symbol
    import MathOptInterface as MOI
    using Logging

    # A code the NLP backends cannot emit (declared unmapped) — the fixture for
    # "a backend one day terminates with a status outside the vocabulary".
    @test MOI.NODE_LIMIT ∈ DirectTrajOpt.Solvers._STATUS_UNMAPPED
    @test_logs (:warn, r"Unmapped solve status") match_mode = :any begin
        @test solve_status_symbol(MOI.NODE_LIMIT) === :unknown
    end
    # Non-MOI garbage degrades the same way.
    @test_logs (:warn, r"Unmapped solve status") match_mode = :any begin
        @test solve_status_symbol("not a real status") === :unknown
    end
end
