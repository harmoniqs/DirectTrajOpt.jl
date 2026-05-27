using DirectTrajOpt
using TestItemRunner

include("test_snippets.jl")

# Exclude `benchmark/` testitems from the main test run — they live in a
# subproject with its own Project.toml (different deps, e.g. HarmoniqsBenchmarks)
# and are exercised by a dedicated workflow.
@run_package_tests filter = ti -> !occursin("/benchmark/", ti.filename)
