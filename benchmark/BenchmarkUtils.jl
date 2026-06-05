# Reporting/display layer for DirectTrajOpt benchmarks.
#
# The benchmark @testitems save BenchmarkResult / MicroBenchmarkResult to JLD2
# under benchmark/results/. This module turns those artifacts into the two
# display surfaces we mirror from CuQuantum.jl:
#
#   1. bench.json  — github-action-benchmark's `customSmallerIsBetter` schema
#      (array of {name, unit, value}). Consumed by benchmark-action in CI to
#      publish a per-commit time-series dashboard to gh-pages + raise
#      regression alerts. Each (benchmark_name, metric) pair is its own series,
#      so series names MUST be stable across commits — benchmark_name already
#      encodes solver + problem size (e.g. "bilinear_N51_ipopt").
#
#   2. A GitHub-flavored markdown table appended to $GITHUB_STEP_SUMMARY (and
#      echoed to stdout) so each CI run shows the numbers inline without
#      downloading the JLD2 artifact.
#
# Kept solver-agnostic: it reads only the HarmoniqsBenchmarks schema, so the
# same module serves Ipopt / MadNLP / any future backend without edits.
module BenchmarkReporting

using HarmoniqsBenchmarks
using JLD2
using Printf

export collect_results, emit_bench_json, emit_markdown, write_report

"""
    collect_results(dir) -> (Vector{BenchmarkResult}, Vector{MicroBenchmarkResult})

Load every `*.jld2` under `dir`, dispatching on the JLD2 top-level key written
by the `save_*` helpers (`results` / `result` / `profile`). Allocation profiles
are skipped here — they have their own report (`report_alloc_profile`). Returns
results sorted by `benchmark_name` for stable output ordering.
"""
function collect_results(dir::AbstractString)
    full = BenchmarkResult[]
    micro = MicroBenchmarkResult[]
    isdir(dir) || return (full, micro)
    for f in sort(readdir(dir))
        endswith(f, ".jld2") || continue
        path = joinpath(dir, f)
        jldopen(path, "r") do io
            if haskey(io, "results")
                append!(full, io["results"])
            elseif haskey(io, "result")
                push!(micro, io["result"])
            end
            # "profile" (AllocProfileResult) intentionally ignored.
        end
    end
    sort!(full, by = r -> (r.benchmark_name, r.solver))
    sort!(micro, by = r -> r.benchmark_name)
    return (full, micro)
end

# Minimal JSON string escaping — benchmark names are effectively identifiers,
# but escape the structural characters so a stray space/quote can't corrupt the
# array the action parses.
_json_escape(s) = replace(string(s), '\\' => "\\\\", '"' => "\\\"")

"""
    emit_bench_json(io, full, micro)

Write the `customSmallerIsBetter` JSON array. Per full-solve result we emit a
wall-time series ("… [wall]", seconds) and an allocation series ("… [alloc]",
bytes); plus, when the solver reported them, an iteration-count series
("… [iters]") and — for convergence suites — the achieved infidelity/objective
("… [infidelity]" / "… [objective]"). Per micro result we emit a median-time
series per evaluated callback ("… / <op> [median]", nanoseconds). Smaller is
better for every series.
"""
function emit_bench_json(
    io::IO,
    full::Vector{BenchmarkResult},
    micro::Vector{MicroBenchmarkResult},
)
    # value is Real: bytes stay Int (printed without exponent), timings Float64.
    entries = Tuple{String,String,Real}[]
    for r in full
        push!(entries, ("$(r.benchmark_name) [wall]", "s", r.wall_time_s))
        push!(entries, ("$(r.benchmark_name) [alloc]", "bytes", r.total_allocations_bytes))
        # Iteration count is meaningful whenever the solver reported it
        # (timing-only runs use the -1 sentinel and are skipped).
        if r.iterations >= 0
            push!(entries, ("$(r.benchmark_name) [iters]", "iterations", r.iterations))
        end
        # Convergence suites carry a criterion — track the achieved
        # infidelity / objective as its own smaller-is-better series.
        c = r.convergence
        if c isa InfidelityConvergence
            push!(
                entries,
                ("$(r.benchmark_name) [infidelity]", "infidelity", c.final_infidelity),
            )
        elseif c isa ObjectiveConvergence
            push!(
                entries,
                ("$(r.benchmark_name) [objective]", "objective", c.final_objective),
            )
        end
    end
    for m in micro
        for (op, eb) in sort(collect(m.eval_benchmarks), by = first)
            push!(entries, ("$(m.benchmark_name) / $(op) [median]", "ns", eb.median_ns))
        end
    end
    print(io, "[")
    for (i, (name, unit, value)) in enumerate(entries)
        i == 1 || print(io, ",")
        print(
            io,
            "{\"name\":\"",
            _json_escape(name),
            "\",\"unit\":\"",
            unit,
            "\",\"value\":",
            value,
            "}",
        )
    end
    print(io, "]")
    return length(entries)
end

# Humanize a byte count to KB/MB/GB for the markdown table (JSON keeps raw bytes
# for machine tracking; humans read the table).
function _human_bytes(n::Integer)
    n < 0 && return "-" * _human_bytes(-n)
    units = ("B", "KB", "MB", "GB", "TB")
    x = Float64(n)
    i = 1
    while x >= 1024 && i < length(units)
        x /= 1024
        i += 1
    end
    return i == 1 ? @sprintf("%d B", n) : @sprintf("%.2f %s", x, units[i])
end

"""
    emit_markdown(io, full, micro)

Write a GitHub-flavored markdown report: a full-solve table (one row per
result, with a convergence column when the result carries a criterion) and a
micro-benchmark table. Safe to call with empty inputs.
"""
function emit_markdown(
    io::IO,
    full::Vector{BenchmarkResult},
    micro::Vector{MicroBenchmarkResult},
)
    println(io, "## DirectTrajOpt benchmark results")
    println(io)
    if isempty(full) && isempty(micro)
        println(io, "_No benchmark results found._")
        return
    end

    if !isempty(full)
        meta = full[1]
        println(
            io,
            "Package `DirectTrajOpt@$(meta.package_version)` · commit `$(meta.commit)` · ",
            "Julia $(meta.julia_version) · runner `$(meta.runner)` · $(meta.n_threads) threads",
        )
        println(io)
        any_conv = any(r -> r.convergence !== nothing, full)
        header = "| Benchmark | Solver | N | dim | Wall (s) | Allocations | Iters | Status |"
        rule = "|---|---|--:|--:|--:|--:|--:|---|"
        if any_conv
            header *= " Converged |"
            rule *= "---|"
        end
        println(io, header)
        println(io, rule)
        for r in full
            row = @sprintf(
                "| `%s` | %s | %d | %d | %.4f | %s | %s | %s |",
                r.benchmark_name,
                r.solver,
                r.N,
                r.state_dim,
                r.wall_time_s,
                _human_bytes(r.total_allocations_bytes),
                r.iterations < 0 ? "—" : string(r.iterations),
                r.solver_status,
            )
            if any_conv
                c = r.convergence
                row *= c === nothing ? " — |" : (converged(c) ? " ✅ |" : " ❌ |")
            end
            println(io, row)
        end
        println(io)
    end

    if !isempty(micro)
        println(io, "### Evaluator micro-benchmarks")
        println(io)
        println(io, "| Benchmark | Callback | Median (ns) | Allocs | Memory |")
        println(io, "|---|---|--:|--:|--:|")
        for m in micro
            for (op, eb) in sort(collect(m.eval_benchmarks), by = first)
                println(
                    io,
                    @sprintf(
                        "| `%s` | `%s` | %.1f | %d | %s |",
                        m.benchmark_name,
                        op,
                        eb.median_ns,
                        eb.allocs,
                        _human_bytes(eb.memory_bytes),
                    )
                )
            end
        end
        println(io)
    end
    return nothing
end

"""
    write_report(results_dir) -> String

Load all results under `results_dir`, write `results_dir/bench.json`, and
append the markdown report to `\$GITHUB_STEP_SUMMARY` when running in CI (always
echoing it to stdout). Returns the path to the written JSON file.
"""
function write_report(results_dir::AbstractString)
    full, micro = collect_results(results_dir)
    mkpath(results_dir)
    json_path = joinpath(results_dir, "bench.json")
    n = open(json_path, "w") do io
        emit_bench_json(io, full, micro)
    end
    @info "Wrote github-action-benchmark JSON" path = json_path series = n

    io = IOBuffer()
    emit_markdown(io, full, micro)
    md = String(take!(io))
    print(stdout, md)
    summary = get(ENV, "GITHUB_STEP_SUMMARY", "")
    if !isempty(summary)
        open(summary, "a") do f
            print(f, md)
        end
    end
    return json_path
end

end # module
