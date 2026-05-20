# Benchmarks

DirectTrajOpt ships a benchmark suite under [`benchmark/`](https://github.com/harmoniqs/DirectTrajOpt.jl/tree/main/benchmark)
that exercises the package under both Ipopt and MadNLP on a shared bilinear
quantum-gate problem: find a pulse sequence ``u(t)`` that steers a qubit state
from ``|0\rangle`` to ``|1\rangle`` under bilinear dynamics

```math
\dot{x}(t) = \left(\omega G_z + u_x(t) G_x + u_y(t) G_y\right) x(t)
```

with ``G_x, G_y, G_z`` the 4×4 real representations of the Pauli generators,
``\omega = 0.1``, and control bound ``|u| \le 0.1``.

!!! note "Example output, not authoritative measurements"
    The tables below show the **shape** of what each benchmark produces, with
    illustrative numbers from one local run. They are not pinned reference
    results — wall-time and allocation figures vary by hardware, BLAS, MUMPS
    build, and Julia version. Don't quote them as the canonical "DirectTrajOpt
    vs MadNLP" comparison. Run the suite yourself on the hardware you care
    about; see [Reproduction](#reproduction) below.

    The benchmark CI workflow on GitHub Actions tracks each solver's wall-time
    and allocation against its own history across commits, which is the only
    apples-to-apples comparison the harness can offer.

## Ipopt vs MadNLP

Same problem (bilinear ``N = 51``, 4D state, 2D control), same initial guess,
same convergence tolerance. Both solvers receive a JIT warmup before timing so
the recorded wall-time reflects steady-state behavior. Metrics captured by
[HarmoniqsBenchmarks.jl](https://github.com/harmoniqs/HarmoniqsBenchmarks.jl)
via `benchmark_solve!`.

### Full solve (bilinear N=51, max_iter=200) — *illustrative*

| Solver | Wall time | Allocations | Objective | Status |
|:-------|:---------:|:-----------:|:---------:|:------:|
| Ipopt  | 8.52 s    | 3.4 GB      | —         | Optimal |
| MadNLP | 5.75 s    | 1.9 GB      | —         | Optimal |

## Evaluator micro-benchmarks

Per-function timings for the MOI evaluator interface on the same bilinear
``N = 51`` problem. Measured with `BenchmarkTools.@benchmark`.

### Per-function timings — *illustrative*

| Function | Median | Allocations | Memory |
|:---------|:------:|:-----------:|:------:|
| `eval_objective` | 0.8 μs | 0 | 0 B |
| `eval_objective_gradient` | 45 μs | 102 | 80 KB |
| `eval_constraint` | 1.2 ms | 5,100 | 4.8 MB |
| `eval_constraint_jacobian` | 3.5 ms | 15,300 | 14 MB |
| `eval_hessian_lagrangian` | 12.7 ms | 73,000 | 68 MB |

`eval_hessian_lagrangian` is typically the dominant per-iteration cost and
the natural optimization target.

## Memory scaling

Both solvers across increasing problem sizes (``N \times \text{state\_dim}``).
Each solver is capped at 50 iterations to measure scaling behavior rather than
convergence. Every ``(N, \text{state\_dim})`` cell uses a deterministic
distinct seed so each data point is a fresh instance; both solvers receive the
same instance per cell to keep that cell's comparison fair.

### Scaling sweep — *illustrative*

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

Each cell is one solve — useful for tracking the slope of each solver vs
itself over time, less useful as a single-shot Ipopt-vs-MadNLP comparison.

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
