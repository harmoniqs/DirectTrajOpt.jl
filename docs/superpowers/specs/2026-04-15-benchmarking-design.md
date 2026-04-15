# HarmoniqsBenchmarks.jl — Cross-Package Benchmarking Infrastructure

**Date:** 2026-04-15
**Status:** Design

## Context

The harmoniqs quantum optimal control stack (DirectTrajOpt, Piccolo, Piccolissimo, Altissimo, Intonato) needs a unified benchmarking system to:

- Compare Ipopt vs MadNLP solver performance on the DirectTrajOpt `feat/madnlp-integration` branch
- Collect statistically robust histograms of key evaluator functions (eval_hessian_lagrangian, eval_constraint_jacobian, etc.) for regression detection
- Profile memory usage and allocations in MadNLP and across all packages, understanding how memory scales with knot points (N), state dimension, and control dimension
- Track allocations in the optimization hot path to drive them toward zero
- Publish version-tagged JLD2 artifacts so labs and enterprises can evaluate problem-size scaling

This is driven by all three active workstreams needing memory/performance benchmarks (MadNLP integration, Altissimo GPU scaling at 1024 state dim, Intonato convergence tracking).

## Architecture

**Approach:** Shared `HarmoniqsBenchmarks.jl` package + per-package `benchmark/` directories + central aggregator repo.

- `HarmoniqsBenchmarks.jl` — lightweight Julia package (own repo in harmoniqs org) providing schema, profiling harness, problem generators, and reporters
- Each downstream package (DirectTrajOpt, Piccolo, Piccolissimo, Altissimo, Intonato) has a `benchmark/` directory with `@testitem`-based benchmarks using `HarmoniqsBenchmarks`
- Central `harmoniqs-benchmarks` repo aggregates artifacts and generates cross-package comparison tables
- Artifacts are JLD2 files stored in CI (GitHub Actions artifact upload), not a live dashboard

## Schema

### BenchmarkResult

```julia
struct BenchmarkResult
    # Identity
    package::String               # "DirectTrajOpt", "Piccolissimo", etc.
    package_version::String       # semver tag
    commit::String                # short SHA
    benchmark_name::String        # "cz_gate_ipopt", "madnlp_scaling_N101_d16"

    # Problem dimensions
    N::Int                        # knot points
    state_dim::Int                # state vector dimension
    control_dim::Int              # number of controls
    n_constraints::Int            # total nonlinear constraints
    n_variables::Int              # total NLP variables

    # Solve metrics
    wall_time_s::Float64
    iterations::Int
    objective_value::Float64
    constraint_violation::Float64
    solver_status::Symbol         # :Optimal, :MaxIter, :Infeasible
    solver::String                # "ipopt", "madnlp", "altissimo"

    # Memory & allocations
    total_allocations_bytes::Int
    total_allocs_count::Int       # number of allocation events
    peak_memory_bytes::Int

    # GC stats
    gc_time_ns::Int
    gc_count::Int
    gc_full_count::Int

    # Solver options snapshot
    solver_options::Dict{Symbol,Any}

    # Metadata
    julia_version::String
    timestamp::DateTime
    runner::String                # "github-actions", "ec2-gpu", "local"
    n_threads::Int
end
```

### MicroBenchmarkResult

```julia
struct MicroBenchmarkResult
    # Identity (same as above)
    package::String
    package_version::String
    commit::String
    benchmark_name::String

    # Problem dimensions
    N::Int
    state_dim::Int
    control_dim::Int

    # Per-function BenchmarkTools results
    # Each value is a serialized BenchmarkTools.Trial containing:
    #   times (ns), gctimes (ns), memory (bytes), allocs (count)
    eval_benchmarks::Dict{Symbol, Any}
    # Keys: :eval_objective, :eval_gradient, :eval_constraint,
    #        :eval_jacobian, :eval_hessian_lagrangian

    # Metadata
    julia_version::String
    timestamp::DateTime
    runner::String
    n_threads::Int
end
```

## Benchmarking Layers

### Layer 1: Micro-benchmarks (Eval Function Histograms)

Use `BenchmarkTools.@benchmark` on individual MOI evaluator methods. This gives statistically robust distributions with proper warmup, plus allocation counts per call.

```julia
@testitem "Evaluator micro-benchmarks: CZ N=51" begin
    using HarmoniqsBenchmarks, BenchmarkTools, Piccolissimo, Piccolo

    prob = build_cz_problem(N=51)
    evaluator, Z_vec = build_evaluator(prob)

    # Pre-allocate output buffers
    g = zeros(n_constraints(evaluator))
    grad = zeros(n_variables(evaluator))
    H = zeros(n_hessian_entries(evaluator))
    J = zeros(n_jacobian_entries(evaluator))
    sigma = 1.0
    mu = ones(n_constraints(evaluator))

    benchmarks = Dict(
        :eval_objective          => @benchmark(MOI.eval_objective($evaluator, $Z_vec)),
        :eval_gradient           => @benchmark(MOI.eval_objective_gradient($evaluator, $grad, $Z_vec)),
        :eval_constraint         => @benchmark(MOI.eval_constraint($evaluator, $g, $Z_vec)),
        :eval_jacobian           => @benchmark(MOI.eval_constraint_jacobian($evaluator, $J, $Z_vec)),
        :eval_hessian_lagrangian => @benchmark(MOI.eval_hessian_lagrangian($evaluator, $H, $Z_vec, $sigma, $mu)),
    )

    save_micro_results("cz_N51_ipopt", benchmarks; prob)
end
```

**Regression detection:** Compare median times and allocation counts across versions. A >10% regression in any eval function on the same problem size flags for review.

### Layer 2: Macro-benchmarks (Full Solves)

Use `@timed` for wall clock + total allocations on `solve!`. Full optimization is not repeatable in the BenchmarkTools sense (each call modifies the problem), so we capture single-run metrics.

```julia
@testitem "CZ gate Ipopt vs MadNLP" begin
    using HarmoniqsBenchmarks, Piccolissimo, Piccolo

    prob = build_cz_problem(N=51)
    result_ipopt = benchmark_solve!(prob, IpoptOptions())

    prob = build_cz_problem(N=51)  # fresh problem
    result_madnlp = benchmark_solve!(prob, MadNLPOptions())

    save_results("cz_gate_comparison", [result_ipopt, result_madnlp])
end
```

### Layer 3: Scaling Studies

Parameterized sweeps over problem dimensions to characterize memory and time growth.

```julia
@testitem "MadNLP memory scaling" begin
    using HarmoniqsBenchmarks, Piccolissimo, Piccolo

    results = BenchmarkResult[]
    for N in [25, 51, 101, 201, 401]
        for state_dim in [4, 8, 16, 32, 64]
            prob = build_bilinear_problem(; N, state_dim, n_controls=2)
            r = benchmark_solve!(prob, MadNLPOptions())
            push!(results, r)
        end
    end
    save_results("madnlp_memory_scaling", results)
end
```

### Layer 4: Allocation Profiling

Tools for tracking down and eliminating allocations in the optimization hot path.

**Profile.Allocs** — captures per-allocation stack traces during a solve:
```julia
@testitem "Allocation profile: CZ solve" begin
    using HarmoniqsBenchmarks, Profile, Piccolissimo, Piccolo

    prob = build_cz_problem(N=51)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1.0 solve!(prob)
    alloc_results = Profile.Allocs.fetch()

    save_alloc_profile("cz_N51_alloc_profile", alloc_results)
    # Visualize locally: using PProf; PProf.Allocs.pprof(alloc_results)
end
```

**AllocCheck.jl** — compile-time zero-allocation enforcement for evaluator hot paths. Can be added as an optional CI check:
```julia
@testitem "Zero-allocation check: evaluator methods" begin
    using AllocCheck, DirectTrajOpt

    # These should be allocation-free once optimized
    @check_allocs MOI.eval_constraint(ev::Evaluator, g::Vector{Float64}, Z::Vector{Float64})
    @check_allocs MOI.eval_constraint_jacobian(ev::Evaluator, J::Vector{Float64}, Z::Vector{Float64})
    @check_allocs MOI.eval_hessian_lagrangian(ev::Evaluator, H::Vector{Float64}, Z::Vector{Float64}, s::Float64, m::Vector{Float64})
end
```

**Per-line tracking** (local development, not CI):
```bash
julia --track-allocation=user --project=benchmark benchmark/benchmarks.jl
# Generates .mem files with per-line allocation counts
```

**Implementation note:** The best allocation profiling approach for the evaluator hot path is TBD. During implementation, spike all three approaches (`Profile.Allocs`, `AllocCheck.jl`, `--track-allocation`) in parallel worktrees against a representative problem (e.g. CZ N=51) to determine which gives the most actionable results for tracking down and eliminating allocations in the MOI eval methods.

## Problem Generators

Deterministic, parameterized problem constructors for reproducibility.

### DirectTrajOpt level
- `build_bilinear_problem(; N=51, state_dim=4, n_controls=2, seed=42)` — random Hermitian system matrices, bilinear integrator + quadratic regularizer
- `build_constrained_problem(; N=51, state_dim=4, n_nonlinear=3, seed=42)` — adds nonlinear knot-point constraints

### Piccolo/Piccolissimo level
- `build_cz_problem(; N=51, integrator=:hermitian_exp)` — 2-qubit CZ gate, exchange-only system (4-level), matches spin-qubit-demo
- `build_cnot_problem(; N=101, integrator=:hermitian_exp)` — 2-qubit CNOT with 3 EDSR drives
- `build_transmon_problem(; levels=3, N=51)` — single-qubit X gate on multi-level transmon

### Altissimo level
- `build_polish_problem(; N=51, state_dim=4)` — pre-solved Ipopt problem ready for Altissimo refinement
- `build_gpu_scaling_problem(; state_dim=1024)` — large-state-dim problem for GPU benchmarking

### Intonato level
- `build_qilc_problem(; N=101, n_paulis=15, J_mismatch=1.3)` — QILC calibration loop with simulated experiment, matches spin-qubit-demo pattern

### Demo-repo-derived problems

The harmoniqs org has several hardware-platform demo repos that provide real-world benchmark problems. During implementation, clone and extract representative problem configurations from:

| Repo | Platform | Typical Dimensions | Key Benchmark |
|------|----------|-------------------|---------------|
| `spin-qubit-demo` | Silicon spin qubits | N=51-101, 4-level, 1-3 drives | CZ, CNOT, QILC calibration |
| `bosonic-demo` | Bosonic cavity QED | Higher Hilbert space dims | Cavity control |
| `nv-center-demo` | NV centers | Spin-1 + nuclear spins | Dark matter sensing pulses |
| `atoms-demo` | Neutral atoms | Rydberg levels | Multi-qubit gates |
| `ions` | Trapped ions | Motional modes + qubits | MS gate, individual addressing |
| `fluxonium-demo` | Fluxonium qubits | Multi-level transmon-like | Single-qubit gates |
| `gkp-stanford` | GKP states | Bosonic Fock space | State preparation |

These provide the "enterprise-scale" problem suite that demonstrates what problem sizes each solver can handle. Extract the system Hamiltonians and problem parameters from each demo, wrap them as generators in `HarmoniqsBenchmarks.problems/`.

All generators use `Random.seed!(seed)` for determinism.

## Harness Functions

### build_evaluator(prob) -> (evaluator, Z_vec)

Extracts the MOI evaluator and initial decision variable vector from a `DirectTrajOptProblem`. Used for micro-benchmarks so individual eval functions can be called directly.

### benchmark_solve!(prob, options; kwargs...) -> BenchmarkResult

```julia
function benchmark_solve!(prob, options; kwargs...)
    GC.gc()
    gc_before = Base.gc_num()

    timed = @timed solve!(prob; options, kwargs...)

    gc_after = Base.gc_num()

    return BenchmarkResult(
        # ... populate from prob metadata, timed, gc delta, options snapshot
    )
end
```

### save_results(name, results) / save_micro_results(name, benchmarks)

Write JLD2 to `benchmark/results/<name>_<commit_sha>.jld2`.

### compare_results(baseline_path, current_path) -> ComparisonTable

Load two result sets and produce a diff table with percent changes, flagging regressions.

## CI Workflow

### Per-package: `.github/workflows/benchmark.yml`

```yaml
name: Benchmarks
on:
  push:
    tags: ['v*']
  workflow_dispatch:
    inputs:
      baseline_tag:
        description: 'Tag to compare against'
        required: false

jobs:
  benchmark:
    runs-on: ubuntu-latest   # free for OSS
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.11'
      - name: Instantiate benchmark env
        run: julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
      - name: Run benchmarks
        run: julia --project=benchmark -t auto -e '
          using TestItemRunner
          @run_package_tests(benchmark)
        '
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-${{ github.ref_name }}-${{ github.sha }}
          path: benchmark/results/
          retention-days: 365

  # GPU/large-scale benchmarks (Altissimo, large N)
  benchmark-gpu:
    if: contains(github.repository, 'Altissimo') || github.event_name == 'workflow_dispatch'
    runs-on: [self-hosted, gpu]   # EC2 runners from CuQuantum.jl setup
    steps:
      # same as above but with CUDA-enabled Julia
```

### Central aggregator: `harmoniqs-benchmarks` repo

Triggered by workflow_dispatch or cron. Downloads latest artifacts from each package repo, generates comparison tables, stores historical archive.

## Package Structure

```
HarmoniqsBenchmarks.jl/
  src/
    HarmoniqsBenchmarks.jl       # module + exports
    schema.jl                     # BenchmarkResult, MicroBenchmarkResult
    harness.jl                    # benchmark_solve!, build_evaluator
    storage.jl                    # save/load JLD2, save_alloc_profile
    report.jl                     # compare_results, regression detection
    problems/
      bilinear.jl                 # DirectTrajOpt-level generators
      quantum_gates.jl            # Piccolo/Piccolissimo-level generators
      polish.jl                   # Altissimo-level generators
      qilc.jl                     # Intonato-level generators
  Project.toml                    # deps: BenchmarkTools, JLD2, Dates
  README.md

# Per downstream package:
DirectTrajOpt.jl/
  benchmark/
    Project.toml                  # [deps] HarmoniqsBenchmarks, BenchmarkTools, TestItems, ...
    benchmarks.jl                 # @testitems: micro, macro, scaling
    results/                      # .gitignored JLD2 output
```

## Verification

1. **Unit test the harness:** `benchmark_solve!` returns a valid `BenchmarkResult` with all fields populated
2. **Run micro-benchmarks locally:** Confirm BenchmarkTools produces histograms for each eval function
3. **Run scaling sweep:** Verify memory grows as expected with N and state_dim
4. **CI dry run:** Trigger workflow_dispatch on DirectTrajOpt, confirm artifact upload
5. **Cross-package comparison:** Run aggregator on two package artifacts, verify comparison table output
6. **Allocation profiling:** Run Profile.Allocs on a solve, verify PProf flamegraph renders

## Scope

**In scope (this design):**
- HarmoniqsBenchmarks.jl package creation
- DirectTrajOpt benchmark suite (Ipopt vs MadNLP, scaling, micro-benchmarks, allocation profiling)
- Piccolissimo benchmark suite (integrate existing benchmarks + new scaling)
- CI workflows for DirectTrajOpt and Piccolissimo
- Aggregator script in harmoniqs-benchmarks repo

**Future work:**
- Altissimo GPU benchmarks (requires CUDA runner validation)
- Intonato convergence benchmarks (requires stable Phase 5)
- Piccolo template benchmarks
- AllocCheck CI gates (after hot paths are optimized)
- Automated regression comments on PRs
