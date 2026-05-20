using DirectTrajOpt
using TestItemRunner

include("test_snippets.jl")

# Tag taxonomy controlling which @testitems run in CI.
#
# Defaults (untagged):
#     Always run. Fast, deterministic, must pass on every PR.
#
# :experimental
#     Known-flaky or environment-sensitive. Excluded by default.
#     Opt in with INCLUDE_EXPERIMENTAL=1 for local diagnosis.
#     Goal: eventually rewrite as deterministic + :robustness pair, then drop the tag.
#
# :robustness
#     Multi-seed sweeps that assert ≥80% of seeds pass within tolerance.
#     Excluded by default because they re-solve a problem many times.
#     Opt in with INCLUDE_ROBUSTNESS=1 (e.g. nightly / scheduled workflows).
#     A regression that drops the true pass rate below ~80% will fail this gate
#     with very high probability (binomial, K=20).
#
# The filter is a single closure so it stays trivial to absorb into a more
# sophisticated upstream filter (e.g. upfront test-item discovery) later.
const INCLUDE_EXPERIMENTAL = haskey(ENV, "INCLUDE_EXPERIMENTAL")
const INCLUDE_ROBUSTNESS = haskey(ENV, "INCLUDE_ROBUSTNESS")

@run_package_tests filter =
    ti -> begin
        tags = get(ti, :tags, Symbol[])
        (INCLUDE_EXPERIMENTAL || !(:experimental in tags)) &&
            (INCLUDE_ROBUSTNESS || !(:robustness in tags))
    end
