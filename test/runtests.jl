using DirectTrajOpt
using TestItemRunner

include("test_snippets.jl")

# Exclude benchmark/ testitems — those run in a separate project environment
# with its own Project.toml (different deps) and are exercised by a dedicated
# workflow. Match the "benchmark" path component exactly so test files like
# foo_benchmark.jl elsewhere in the tree aren't accidentally skipped.
@run_package_tests filter = ti -> !("benchmark" in splitpath(ti.filename))
