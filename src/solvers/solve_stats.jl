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
