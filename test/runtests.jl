using DirectTrajOpt
using TestItemRunner

include("test_snippets.jl")

# Exclude `benchmark/` testitems — they live in a subproject with its own
# Project.toml (different deps, e.g. HarmoniqsBenchmarks) and are exercised by a
# dedicated workflow. Match the "benchmark" path component exactly so test
# files like foo_benchmark.jl elsewhere in the tree aren't accidentally skipped.
@run_package_tests filter = ti -> !("benchmark" in splitpath(ti.filename))
