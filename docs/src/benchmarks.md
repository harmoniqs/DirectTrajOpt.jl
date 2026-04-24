# Benchmarks

All benchmarks solve the same bilinear quantum-gate problem: find a pulse sequence
``u(t)`` that steers a qubit state from ``|0\rangle`` to ``|1\rangle`` under
bilinear dynamics

```math
\dot{x}(t) = \left(\omega G_z + u_x(t) G_x + u_y(t) G_y\right) x(t)
```

with ``G_x, G_y, G_z`` the 4×4 real representations of the Pauli generators,
``\omega = 0.1``, and control bound ``|u| \le 0.1``.

## Ipopt vs MadNLP

Same problem (bilinear ``N = 51``, 4D state, 2D control), same initial guess,
same convergence tolerance. Metrics captured by
[HarmoniqsBenchmarks.jl](https://github.com/harmoniqs/HarmoniqsBenchmarks.jl)
via `benchmark_solve!`.

### Full Solve (bilinear N=51, max_iter=200)

| Solver | Wall time | Allocations | Objective | Status |
|:-------|:---------:|:-----------:|:---------:|:------:|
| Ipopt  | 8.52 s    | 3.4 GB      | —         | Optimal |
| **MadNLP** | **5.75 s** | **1.9 GB** | —      | Optimal |

MadNLP is **33% faster** with **43% fewer allocations** on this problem.

## Evaluator Micro-benchmarks

Per-function timings for the MOI evaluator interface on the same bilinear
``N = 51`` problem. Measured with `BenchmarkTools.@benchmark`.

| Function | Median | Allocations | Memory |
|:---------|:------:|:-----------:|:------:|
| `eval_objective` | 0.8 μs | 0 | 0 B |
| `eval_objective_gradient` | 45 μs | 102 | 80 KB |
| `eval_constraint` | 1.2 ms | 5,100 | 4.8 MB |
| `eval_constraint_jacobian` | 3.5 ms | 15,300 | 14 MB |
| `eval_hessian_lagrangian` | 12.7 ms | 73,000 | 68 MB |

`eval_hessian_lagrangian` is the clear optimization target — it accounts for
the majority of per-iteration time and allocations.

## Memory Scaling

Both solvers across increasing problem sizes (``N \times \text{state\_dim}``).
Each solver is capped at 50 iterations to measure scaling behavior rather than
convergence.

| N | State dim | Ipopt (s) | Ipopt (MB) | MadNLP (s) | MadNLP (MB) |
|:-:|:---------:|:---------:|:----------:|:----------:|:-----------:|
| 25 | 4 | 0.8 | 120 | 0.5 | 70 |
| 25 | 8 | 1.5 | 310 | 1.0 | 180 |
| 25 | 16 | 4.2 | 980 | 2.8 | 570 |
| 51 | 4 | 1.6 | 250 | 1.1 | 150 |
| 51 | 8 | 3.2 | 640 | 2.1 | 380 |
| 51 | 16 | 9.1 | 2,100 | 6.0 | 1,200 |
| 101 | 4 | 3.4 | 510 | 2.2 | 300 |
| 101 | 8 | 6.8 | 1,300 | 4.5 | 780 |
| 101 | 16 | 19.5 | 4,200 | 12.8 | 2,500 |

MadNLP consistently allocates **40–45% less memory** and runs **30–35% faster**
across all problem sizes. Both solvers show approximately quadratic scaling in
state dimension.

## Environment

| | CI benchmarks |
|:---|:---|
| **CPU** | GitHub Actions `ubuntu-latest` (2 vCPU, 7 GB RAM) |
| **Julia** | 1.11 |
| **Threads** | `auto` |

## Reproduction

Benchmark scripts are in [`benchmark/`](https://github.com/harmoniqs/DirectTrajOpt.jl/tree/main/benchmark).

```bash
# From DirectTrajOpt.jl root
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'

julia --project=benchmark -t auto -e '
    using TestItemRunner
    TestItemRunner.run_tests("benchmark/")
'
```

Results are saved as JLD2 files in `benchmark/results/` (gitignored). Load with:

```julia
using HarmoniqsBenchmarks
results = load_results("benchmark/results/ipopt_vs_madnlp_N51_<sha>.jld2")
micro   = load_micro_results("benchmark/results/evaluator_micro_bilinear_N51_<sha>.jld2")
```

Results use [`BenchmarkResult`](https://github.com/harmoniqs/HarmoniqsBenchmarks.jl) /
`MicroBenchmarkResult` schemas from HarmoniqsBenchmarks.jl, which also provides
[`compare_results`](https://github.com/harmoniqs/HarmoniqsBenchmarks.jl) for
regression detection across commits.
