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

## Continuous tracking

Every benchmark CI run post-processes its saved `BenchmarkResult` artifacts
(`benchmark/report.jl`) into two display surfaces, mirroring the approach in
[CuQuantum.jl](https://github.com/harmoniqs/CuQuantum.jl):

- **Live dashboard.** A `customSmallerIsBetter` JSON (`benchmark/results/bench.json`)
  is published by [`github-action-benchmark`](https://github.com/benchmark-action/github-action-benchmark)
  to the [**`bench/` dashboard on gh-pages**](https://harmoniqs.github.io/DirectTrajOpt.jl/bench/).
  Each `(benchmark, metric)` pair — e.g. `bilinear_N51_ipopt [wall]`,
  `bilinear_N51_madnlp [alloc]` — is tracked as its own per-commit time series.
- **Regression alerts.** Any series that regresses by more than **120 %** versus
  its history raises a comment on the offending commit/PR. Alerts never fail the
  build (`fail-on-alert: false`); they flag, they don't block. The series are
  only saved/pushed on `main`, so branch and PR runs render a comparison without
  polluting the published history.
- **Per-run job summary.** The same numbers are written to the Actions run's
  job summary as a markdown table, so each run shows its results inline without
  downloading the JLD2 artifact.

## Ipopt vs MadNLP

Same problem (bilinear ``N = 51``, 4D state, 2D control), same initial guess,
same convergence tolerance. Both solvers receive a JIT warmup before timing so
the recorded wall-time reflects steady-state behavior. Metrics captured by
[HarmoniqsBenchmarks.jl](https://github.com/harmoniqs/HarmoniqsBenchmarks.jl)
via `benchmark_solve!`.

### Full solve (bilinear N=51, max_iter=200)

Snapshot from commit `dd0beb4` on GH Actions `ubuntu-latest` (Julia 1.11, 2 vCPU).
Numbers vary by hardware/BLAS/MUMPS build — the [live dashboard](https://harmoniqs.github.io/DirectTrajOpt.jl/bench/) is the source of truth.

| Solver | Wall time | Allocations |
|:-------|:---------:|:-----------:|
| Ipopt  | 0.617 s   | 1.37 GiB    |
| MadNLP | 0.401 s   | 0.94 GiB    |

`Allocations` is **total** (cumulative transient) allocation, not peak RSS — it's
dominated by `ForwardDiff` Hessian/Jacobian buffers (corroborated by the
allocation profile, whose 1 %-sampled total scales back to the same ~1.4 GiB).

## Evaluator micro-benchmarks

Per-function timings for the MOI evaluator interface on the same bilinear
``N = 51`` problem. Measured with `BenchmarkTools.@benchmark`.

### Per-function timings

Snapshot from commit `dd0beb4` (GH Actions `ubuntu-latest`, Julia 1.11); see the
[live dashboard](https://harmoniqs.github.io/DirectTrajOpt.jl/bench/) for current values.

| Function | Median | Allocations | Memory |
|:---------|:------:|:-----------:|:------:|
| `eval_objective` | 185 μs | 3,373 | 159 KB |
| `eval_objective_gradient` | 209 μs | 4,288 | 197 KB |
| `eval_constraint` | 1.0 ms | 16,743 | 999 KB |
| `eval_constraint_jacobian` | 2.0 ms | 27,043 | 3.4 MB |
| `eval_hessian_lagrangian` | 22.3 ms | 70,968 | 68.9 MB |

`eval_hessian_lagrangian` is the dominant per-iteration cost (~10× the Jacobian)
and the natural optimization target — consistent with the allocation profile,
where `ForwardDiff` Hessian buffers top the breakdown.

## Memory scaling

Both solvers across increasing problem sizes (``N \times \text{state\_dim}``).
Each solver is capped at 50 iterations to measure scaling behavior rather than
convergence. Every ``(N, \text{state\_dim})`` cell runs ``K = 3`` random
instances (deterministic distinct seeds) and the table shows the **median** wall
time and allocation total across those seeds — single-shot timings on random
instances are noisy enough that one degenerate seed can dominate a cell. Both
solvers receive the same instance per (cell, seed) so per-seed Ipopt-vs-MadNLP
comparisons stay fair; only the choice of instance varies across the K samples.

The per-seed `BenchmarkResult`s are all saved to the JLD2 artifact, so the
raw distribution behind each median cell is available for downstream analysis.

### Scaling sweep

Median over ``K = 3`` seeds per cell, commit `dd0beb4` (GH Actions
`ubuntu-latest`, Julia 1.11). Allocations are **GB** (total transient, not peak);
see the [live dashboard](https://harmoniqs.github.io/DirectTrajOpt.jl/bench/) for
the per-commit trend.

| N | State dim | Ipopt (s) | Ipopt (GB) | MadNLP (s) | MadNLP (GB) |
|:-:|:---------:|:---------:|:----------:|:----------:|:-----------:|
| 25 | 4 | 0.02 | 0.0 | 0.86 | 1.6 |
| 25 | 8 | 0.01 | 0.0 | 3.78 | 7.4 |
| 25 | 16 | 0.57 | 1.0 | 26.68 | 48.8 |
| 51 | 4 | 2.59 | 3.9 | 1.71 | 3.2 |
| 51 | 8 | 7.00 | 14.4 | 7.48 | 15.4 |
| 51 | 16 | 57.13 | 102.6 | 51.79 | 98.4 |
| 101 | 4 | 3.51 | 6.6 | 3.02 | 6.0 |
| 101 | 8 | 14.93 | 30.9 | 13.19 | 28.7 |
| 101 | 16 | 114.43 | 199.0 | 98.51 | 193.2 |

The near-zero Ipopt cells at ``N = 25`` (dim 4, 8) are real: on those small
random instances Ipopt's interior-point method hits an acceptable point almost
immediately, whereas MadNLP still runs its full iteration budget. At larger
sizes the two are comparable, with MadNLP modestly faster at the largest cell.

Each cell is the median over ``K = 3`` solves on independent random
instances — most useful for tracking the slope of each solver vs itself
over time, less useful as an instance-by-instance Ipopt-vs-MadNLP
comparison since the underlying problems differ between cells.

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

To regenerate the dashboard JSON (`bench.json`) and a markdown summary from the
saved artifacts — exactly what CI runs after the suite:

```bash
julia --project=benchmark benchmark/report.jl
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
