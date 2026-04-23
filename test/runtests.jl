using DirectTrajOpt
using TestItemRunner

include("test_snippets.jl")

# Run all testitem tests in package
@run_package_tests
