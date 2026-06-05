# Display layer for the allocation-profile suite.
#
#     julia --project=benchmark benchmark/alloc_report.jl
#
# Allocation profiles are a different artifact than solve results
# (AllocProfileResult, saved under benchmark/results/allocs/), so they get their
# own reporter rather than the shared BenchmarkReporting module. It produces:
#
#   1. benchmark/results/allocs/bench.json — github-action-benchmark
#      `customSmallerIsBetter` series tracking each solver's total sampled
#      allocations + sample count over commits (published to /bench-alloc).
#   2. A $GITHUB_STEP_SUMMARY markdown report: a totals table plus the full
#      `report_alloc_profile` top-types/leaves/frames breakdown per solver,
#      so the detail is visible in the run summary without digging through logs.
using HarmoniqsBenchmarks
using Printf

allocs_dir = joinpath(@__DIR__, "results", "allocs")

profiles = AllocProfileResult[]
if isdir(allocs_dir)
    for f in sort(readdir(allocs_dir))
        endswith(f, ".jld2") || continue
        push!(profiles, load_alloc_profile(joinpath(allocs_dir, f)))
    end
end
sort!(profiles, by = p -> (p.benchmark_name, p.solver))

_json_escape(s) = replace(string(s), '\\' => "\\\\", '"' => "\\\"")

# customSmallerIsBetter JSON — sampled totals are a consistent regression signal
# across commits (same sample_rate), so we track them raw.
mkpath(allocs_dir)
json_path = joinpath(allocs_dir, "bench.json")
open(json_path, "w") do io
    print(io, "[")
    first = true
    for p in profiles
        for (suffix, unit, value) in (
            ("alloc-total", "bytes", p.total_bytes),
            ("alloc-count", "allocs", p.total_count),
        )
            first || print(io, ",")
            first = false
            print(
                io,
                "{\"name\":\"",
                _json_escape("$(p.benchmark_name) [$(suffix)]"),
                "\",\"unit\":\"",
                unit,
                "\",\"value\":",
                value,
                "}",
            )
        end
    end
    print(io, "]")
end
@info "Wrote alloc-profile JSON" path = json_path series = 2 * length(profiles)

# Markdown report: totals table + full per-profile breakdown.
io = IOBuffer()
println(io, "## DirectTrajOpt allocation profile")
println(io)
if isempty(profiles)
    println(io, "_No allocation profiles found._")
else
    meta = profiles[1]
    println(
        io,
        "Package `DirectTrajOpt` · commit `$(meta.commit)` · Julia $(meta.julia_version) · runner `$(meta.runner)`",
    )
    println(io)
    println(io, "| Benchmark | Solver | Sampled bytes | Sampled allocs | Sample rate |")
    println(io, "|---|---|--:|--:|--:|")
    for p in profiles
        println(
            io,
            @sprintf(
                "| `%s` | %s | %d | %d | %g |",
                p.benchmark_name,
                p.solver,
                p.total_bytes,
                p.total_count,
                p.sample_rate,
            )
        )
    end
    println(io)
    for p in profiles
        println(io, "<details><summary><code>$(p.benchmark_name)</code> / $(p.solver) — top allocations</summary>")
        println(io)
        println(io, "```")
        report_alloc_profile(p; io = io, k_types = 10, k_leaves = 15, k_frames = 15)
        println(io, "```")
        println(io)
        println(io, "</details>")
        println(io)
    end
end
md = String(take!(io))
print(stdout, md)
summary = get(ENV, "GITHUB_STEP_SUMMARY", "")
isempty(summary) || open(f -> print(f, md), summary, "a")
