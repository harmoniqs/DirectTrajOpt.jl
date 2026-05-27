using DirectTrajOpt
using TestItemRunner

include("test_snippets.jl")

# Exclude benchmark/ testitems — those run in a separate project environment
# with HarmoniqsBenchmarks.jl as a dependency. Match the "benchmark" path
# component exactly so a future test file like foo_benchmark.jl isn't
# accidentally skipped.
@run_package_tests filter = ti -> !("benchmark" in splitpath(ti.filename))
