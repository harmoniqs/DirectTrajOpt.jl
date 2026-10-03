# Post-process the convergence suite's saved artifacts into display surfaces.
#
#     julia --project=benchmark/convergence benchmark/convergence/report.jl
#
# Reuses the shared BenchmarkReporting module (benchmark/BenchmarkUtils.jl):
# writes benchmark/convergence/results/bench.json (github-action-benchmark
# schema, incl. per-result [iters]/[infidelity] series) and appends a markdown
# table to $GITHUB_STEP_SUMMARY when run in CI.
using HarmoniqsBenchmarks
include(joinpath(@__DIR__, "..", "BenchmarkUtils.jl"))
using .BenchmarkReporting

BenchmarkReporting.write_report(joinpath(@__DIR__, "results"))
