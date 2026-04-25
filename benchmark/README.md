# DirectTrajOpt Benchmarks

Benchmark suite for DirectTrajOpt.jl comparing Ipopt and MadNLP solver performance.

For results and analysis, see the [Benchmarks page](https://docs.harmoniqs.co/DirectTrajOpt.jl/dev/benchmarks/) in the documentation.

## Running locally

```bash
# From DirectTrajOpt.jl root
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'

julia --project=benchmark -t auto -e '
    using TestItemRunner
    TestItemRunner.run_tests("benchmark/")
'
```

Artifacts are saved as JLD2 files in `benchmark/results/` (gitignored).

## Benchmark suites

- **Evaluator micro-benchmarks** — `BenchmarkTools.@benchmark` timings for each MOI eval function (objective, gradient, constraint, jacobian, hessian_lagrangian) on bilinear N=51
- **Ipopt vs MadNLP** — full solve comparison on bilinear N=51
- **Memory scaling study** — N ∈ {25, 51, 101} × state_dim ∈ {4, 8, 16}

## Schema

Results use `BenchmarkResult` / `MicroBenchmarkResult` from [HarmoniqsBenchmarks.jl](https://github.com/harmoniqs/HarmoniqsBenchmarks.jl).

Load with:
```julia
using HarmoniqsBenchmarks
results = load_results("benchmark/results/ipopt_vs_madnlp_N51_<sha>.jld2")
micro   = load_micro_results("benchmark/results/evaluator_micro_bilinear_N51_<sha>.jld2")
```

## Regression detection

```julia
using HarmoniqsBenchmarks
baseline = load_results("benchmark/results/memory_scaling_<old_sha>.jld2")
current  = load_results("benchmark/results/memory_scaling_<new_sha>.jld2")
rows = compare_results(baseline, current; regression_threshold=10.0)
```
