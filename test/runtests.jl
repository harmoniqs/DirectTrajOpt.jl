using DirectTrajOpt
using TestItemRunner


# Run all testitem tests in package
# Filter out experimental tests unless INCLUDE_EXPERIMENTAL environment variable is set
if !haskey(ENV, "INCLUDE_EXPERIMENTAL")
    @run_package_tests filter=ti -> !(:experimental in get(ti, :tags, []))
else
    @run_package_tests
end
