using DirectTrajOpt
using TestItemRunner


# Exclude benchmark/ testitems — those run in a separate project environment
@run_package_tests filter=ti -> !contains(ti.filename, "benchmark")
