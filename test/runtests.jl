using DirectTrajOpt
using TestItemRunner

include("test_snippets.jl")

# Exclude benchmark/ testitems — those run in a separate project environment.
# Match the "benchmark" path component exactly so test files like
# foo_benchmark.jl elsewhere in the tree aren't accidentally skipped.
@run_package_tests filter = ti -> !("benchmark" in splitpath(ti.filename))
