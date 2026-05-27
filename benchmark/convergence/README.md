# DirectTrajOpt — Convergence Benchmarks

Convergence-quality benchmarks built on HarmoniqsBenchmarks.jl v0.2.0's new
convergence API (`InfidelityConvergence`, `ipopt_capture`,
`compare_convergence`). Each `@testitem` runs a small X-gate state-transfer
on Ipopt or MadNLP and asserts the result meets a problem-specific success
bar before saving a JLD2 artifact under `benchmark/convergence/results/`.

## Running locally

```bash
# from DirectTrajOpt.jl root
julia --project=benchmark/convergence -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/convergence -e '
    using TestItemRunner
    @run_package_tests filter = ti -> occursin("convergence", ti.name)
'
```

## What's covered

- **X gate convergence: Ipopt** — uses `ipopt_capture()` to grab final
  `iter_count` + `inf_pr`, builds an `InfidelityConvergence`, passes it
  through `benchmark_solve!` to populate `BenchmarkResult.convergence`.
- **X gate convergence: MadNLP** — same problem, MadNLP solver. No capture
  hook yet, so `primal_infeasibility` is taken from the post-solve
  evaluator's `constraint_violation`.

Atoms / spin-qubit / bosonic demo problems land in separate follow-up PRs.
