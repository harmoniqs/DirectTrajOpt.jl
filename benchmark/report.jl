# Post-process saved benchmark artifacts into display surfaces.
#
# Run after the benchmark suite (which writes JLD2 to benchmark/results/):
#
#     julia --project=benchmark benchmark/report.jl
#
# Writes benchmark/results/bench.json (github-action-benchmark schema) and
# appends a markdown table to $GITHUB_STEP_SUMMARY when run in CI.
using HarmoniqsBenchmarks
include(joinpath(@__DIR__, "BenchmarkUtils.jl"))
using .BenchmarkReporting

BenchmarkReporting.write_report(joinpath(@__DIR__, "results"))
