# HarmoniqsBenchmarks.jl + DirectTrajOpt Benchmark Suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a shared benchmarking package (`HarmoniqsBenchmarks.jl`) and wire up the first benchmark suite in DirectTrajOpt.jl comparing Ipopt vs MadNLP, with micro-benchmarks, full-solve benchmarks, and memory scaling studies.

**Architecture:** HarmoniqsBenchmarks.jl provides schema types, a profiling harness, and JLD2 storage/comparison. DirectTrajOpt.jl's `benchmark/` directory contains `@testitem`-based benchmarks that use the shared harness. Both Ipopt and MadNLP benchmarks use the same shared `Evaluator` (in `src/solvers/evaluator.jl`), so micro-benchmarks are solver-agnostic while macro-benchmarks compare the two solver backends.

**Tech Stack:** Julia 1.11+, BenchmarkTools.jl, JLD2.jl, TestItems/TestItemRunner, MathOptInterface

**Spec:** `docs/superpowers/specs/2026-04-15-benchmarking-design.md`

---

## File Structure

### New repo: `HarmoniqsBenchmarks.jl` (at `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/`)

| File | Responsibility |
|------|---------------|
| `Project.toml` | Package metadata + deps (BenchmarkTools, JLD2, Dates, DirectTrajOpt, MathOptInterface, NamedTrajectories) |
| `src/HarmoniqsBenchmarks.jl` | Module definition + exports |
| `src/schema.jl` | `BenchmarkResult`, `MicroBenchmarkResult`, `EvalBenchmark` structs |
| `src/harness.jl` | `build_evaluator`, `benchmark_solve!`, GC/allocation capture |
| `src/storage.jl` | `save_results`, `save_micro_results`, `load_results`, `load_micro_results` |
| `src/report.jl` | `compare_results` — diff tables + regression flagging |
| `test/runtests.jl` | Tests for all of the above |

### Modified repo: `DirectTrajOpt.jl` (benchmark directory)

| File | Responsibility |
|------|---------------|
| `benchmark/Project.toml` | Benchmark env deps (HarmoniqsBenchmarks, BenchmarkTools, TestItems, MadNLP) |
| `benchmark/benchmarks.jl` | `@testitem` definitions: micro, macro, scaling |
| `benchmark/.gitignore` | Ignore `results/` directory |

---

## Task 1: Create HarmoniqsBenchmarks.jl Project Skeleton

**Files:**
- Create: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/Project.toml`
- Create: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/HarmoniqsBenchmarks.jl`

- [ ] **Step 1: Initialize the package directory**

```bash
mkdir -p /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src
mkdir -p /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/test
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
git init
```

- [ ] **Step 2: Create Project.toml**

```toml
name = "HarmoniqsBenchmarks"
uuid = "GENERATE_UUID"
version = "0.1.0"
authors = ["harmoniqs contributors"]

[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
DirectTrajOpt = "c823fa1f-8872-4af5-b810-2b9b72bbbf56"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
NamedTrajectories = "538bc3a1-5ab9-4fc3-b776-35ca1e893e08"

[compat]
BenchmarkTools = "1.6"
Dates = "1.10, 1.11, 1.12"
DirectTrajOpt = "0.8"
JLD2 = "0.5"
MathOptInterface = "1.49"
NamedTrajectories = "0.8"
julia = "1.10, 1.11, 1.12"
```

Generate the UUID with: `using UUIDs; uuid4()`

- [ ] **Step 3: Create module stub**

```julia
# src/HarmoniqsBenchmarks.jl
module HarmoniqsBenchmarks

end
```

- [ ] **Step 4: Dev-install dependencies and verify the package loads**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e '
    using Pkg
    Pkg.develop(path="../DirectTrajOpt.jl")
    Pkg.develop(path="../NamedTrajectories.jl")
    Pkg.instantiate()
    using HarmoniqsBenchmarks
    println("Package loads OK")
'
```

Expected: "Package loads OK"

- [ ] **Step 5: Commit**

```bash
git add Project.toml src/HarmoniqsBenchmarks.jl
git commit -m "feat: initialize HarmoniqsBenchmarks.jl package skeleton"
```

---

## Task 2: Implement Schema Types

**Files:**
- Create: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/schema.jl`
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/HarmoniqsBenchmarks.jl`
- Create: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/test/runtests.jl`

- [ ] **Step 1: Write tests for schema types**

```julia
# test/runtests.jl
using Test
using HarmoniqsBenchmarks
using Dates

@testset "HarmoniqsBenchmarks" begin

@testset "Schema" begin
    @testset "EvalBenchmark construction" begin
        eb = EvalBenchmark(
            times_ns = [100.0, 110.0, 105.0],
            gctimes_ns = [0.0, 0.0, 5.0],
            memory_bytes = 1024,
            allocs = 3,
        )
        @test eb.median_ns == 105.0
        @test eb.min_ns == 100.0
        @test 104.0 < eb.mean_ns < 106.0
    end

    @testset "BenchmarkResult construction" begin
        r = BenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "abc1234",
            benchmark_name = "test_bench",
            N = 51,
            state_dim = 4,
            control_dim = 2,
            n_constraints = 200,
            n_variables = 765,
            wall_time_s = 1.5,
            iterations = 42,
            objective_value = 0.001,
            constraint_violation = 1e-8,
            solver_status = :Optimal,
            solver = "ipopt",
            total_allocations_bytes = 1_000_000,
            total_allocs_count = 500,
            gc_time_ns = 10_000,
            gc_count = 2,
            gc_full_count = 0,
            solver_options = Dict{Symbol,Any}(:tol => 1e-8, :max_iter => 1000),
            julia_version = string(VERSION),
            timestamp = now(),
            runner = "local",
            n_threads = 1,
        )
        @test r.package == "DirectTrajOpt"
        @test r.solver_status == :Optimal
    end

    @testset "MicroBenchmarkResult construction" begin
        eb = EvalBenchmark(
            times_ns = [100.0],
            gctimes_ns = [0.0],
            memory_bytes = 0,
            allocs = 0,
        )
        mr = MicroBenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "abc1234",
            benchmark_name = "micro_test",
            N = 51,
            state_dim = 4,
            control_dim = 2,
            eval_benchmarks = Dict{Symbol,EvalBenchmark}(
                :eval_objective => eb,
            ),
            julia_version = string(VERSION),
            timestamp = now(),
            runner = "local",
            n_threads = 1,
        )
        @test mr.eval_benchmarks[:eval_objective].min_ns == 100.0
    end
end

end # HarmoniqsBenchmarks testset
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL — `EvalBenchmark` not defined

- [ ] **Step 3: Implement schema types**

```julia
# src/schema.jl
using Dates
using Statistics: median, mean

struct EvalBenchmark
    times_ns::Vector{Float64}
    gctimes_ns::Vector{Float64}
    memory_bytes::Int
    allocs::Int
    # Derived stats (computed at construction)
    median_ns::Float64
    min_ns::Float64
    mean_ns::Float64
end

function EvalBenchmark(;
    times_ns::Vector{Float64},
    gctimes_ns::Vector{Float64},
    memory_bytes::Int,
    allocs::Int,
)
    return EvalBenchmark(
        times_ns,
        gctimes_ns,
        memory_bytes,
        allocs,
        median(times_ns),
        minimum(times_ns),
        mean(times_ns),
    )
end

struct BenchmarkResult
    # Identity
    package::String
    package_version::String
    commit::String
    benchmark_name::String
    # Problem dimensions
    N::Int
    state_dim::Int
    control_dim::Int
    n_constraints::Int
    n_variables::Int
    # Solve metrics
    wall_time_s::Float64
    iterations::Int
    objective_value::Float64
    constraint_violation::Float64
    solver_status::Symbol
    solver::String
    # Memory & allocations
    total_allocations_bytes::Int
    total_allocs_count::Int
    gc_time_ns::Int
    gc_count::Int
    gc_full_count::Int
    # Solver options snapshot
    solver_options::Dict{Symbol,Any}
    # Metadata
    julia_version::String
    timestamp::DateTime
    runner::String
    n_threads::Int
end

struct MicroBenchmarkResult
    package::String
    package_version::String
    commit::String
    benchmark_name::String
    N::Int
    state_dim::Int
    control_dim::Int
    eval_benchmarks::Dict{Symbol,EvalBenchmark}
    julia_version::String
    timestamp::DateTime
    runner::String
    n_threads::Int
end
```

- [ ] **Step 4: Update module to include schema and export types**

```julia
# src/HarmoniqsBenchmarks.jl
module HarmoniqsBenchmarks

export EvalBenchmark, BenchmarkResult, MicroBenchmarkResult

include("schema.jl")

end
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
git add src/schema.jl src/HarmoniqsBenchmarks.jl test/runtests.jl
git commit -m "feat: add BenchmarkResult, MicroBenchmarkResult, EvalBenchmark schema types"
```

---

## Task 3: Implement JLD2 Storage

**Files:**
- Create: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/storage.jl`
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/HarmoniqsBenchmarks.jl`
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/test/runtests.jl`

- [ ] **Step 1: Add storage tests**

Append to `test/runtests.jl`, inside the top-level `@testset "HarmoniqsBenchmarks"`:

```julia
@testset "Storage" begin
    mktempdir() do dir
        r = BenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "abc1234",
            benchmark_name = "storage_test",
            N = 51, state_dim = 4, control_dim = 2,
            n_constraints = 200, n_variables = 765,
            wall_time_s = 1.5, iterations = 42,
            objective_value = 0.001, constraint_violation = 1e-8,
            solver_status = :Optimal, solver = "ipopt",
            total_allocations_bytes = 1_000_000, total_allocs_count = 500,
            gc_time_ns = 10_000, gc_count = 2, gc_full_count = 0,
            solver_options = Dict{Symbol,Any}(:tol => 1e-8),
            julia_version = string(VERSION),
            timestamp = now(), runner = "local", n_threads = 1,
        )

        path = save_results(dir, "test_bench", [r])
        @test isfile(path)
        @test endswith(path, ".jld2")

        loaded = load_results(path)
        @test length(loaded) == 1
        @test loaded[1].package == "DirectTrajOpt"
        @test loaded[1].wall_time_s == 1.5
        @test loaded[1].solver_options[:tol] == 1e-8
    end

    mktempdir() do dir
        eb = EvalBenchmark(
            times_ns = [100.0, 110.0],
            gctimes_ns = [0.0, 0.0],
            memory_bytes = 512, allocs = 1,
        )
        mr = MicroBenchmarkResult(
            package = "DirectTrajOpt",
            package_version = "0.8.10",
            commit = "abc1234",
            benchmark_name = "micro_storage_test",
            N = 51, state_dim = 4, control_dim = 2,
            eval_benchmarks = Dict(:eval_objective => eb),
            julia_version = string(VERSION),
            timestamp = now(), runner = "local", n_threads = 1,
        )

        path = save_micro_results(dir, "micro_test", mr)
        @test isfile(path)

        loaded = load_micro_results(path)
        @test loaded.benchmark_name == "micro_storage_test"
        @test loaded.eval_benchmarks[:eval_objective].min_ns == 100.0
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL — `save_results` not defined

- [ ] **Step 3: Implement storage functions**

```julia
# src/storage.jl
using JLD2

"""
    save_results(dir, name, results::Vector{BenchmarkResult}) -> String

Save benchmark results to a JLD2 file in `dir`. Returns the file path.
"""
function save_results(dir::String, name::String, results::Vector{BenchmarkResult})
    mkpath(dir)
    commit = isempty(results) ? "unknown" : results[1].commit
    filename = "$(name)_$(commit).jld2"
    path = joinpath(dir, filename)
    JLD2.jldsave(path; results=results)
    return path
end

"""
    load_results(path) -> Vector{BenchmarkResult}

Load benchmark results from a JLD2 file.
"""
function load_results(path::String)
    return JLD2.load(path, "results")
end

"""
    save_micro_results(dir, name, result::MicroBenchmarkResult) -> String

Save micro-benchmark results to a JLD2 file in `dir`. Returns the file path.
"""
function save_micro_results(dir::String, name::String, result::MicroBenchmarkResult)
    mkpath(dir)
    filename = "$(name)_$(result.commit).jld2"
    path = joinpath(dir, filename)
    JLD2.jldsave(path; result=result)
    return path
end

"""
    load_micro_results(path) -> MicroBenchmarkResult

Load micro-benchmark results from a JLD2 file.
"""
function load_micro_results(path::String)
    return JLD2.load(path, "result")
end
```

- [ ] **Step 4: Update module**

```julia
# src/HarmoniqsBenchmarks.jl
module HarmoniqsBenchmarks

export EvalBenchmark, BenchmarkResult, MicroBenchmarkResult
export save_results, load_results, save_micro_results, load_micro_results

include("schema.jl")
include("storage.jl")

end
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
git add src/storage.jl src/HarmoniqsBenchmarks.jl test/runtests.jl
git commit -m "feat: add JLD2 save/load for BenchmarkResult and MicroBenchmarkResult"
```

---

## Task 4: Implement build_evaluator Harness

**Files:**
- Create: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/harness.jl`
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/HarmoniqsBenchmarks.jl`
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/test/runtests.jl`

- [ ] **Step 1: Add test for build_evaluator**

Append to `test/runtests.jl`, inside top-level testset:

```julia
@testset "Harness" begin
    using DirectTrajOpt
    using NamedTrajectories
    using SparseArrays
    using ExponentialAction
    using MathOptInterface
    const MOI = MathOptInterface

    # Build a simple bilinear problem (same as DirectTrajOpt test_utils.jl)
    N = 10; Δt = 0.1; u_bound = 0.1; ω = 0.1
    Gx = sparse(Float64[0 0 0 1; 0 0 1 0; 0 -1 0 0; -1 0 0 0])
    Gy = sparse(Float64[0 -1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 1 0])
    Gz = sparse(Float64[0 0 1 0; 0 0 0 -1; -1 0 0 0; 0 1 0 0])
    G(u) = ω * Gz + u[1] * Gx + u[2] * Gy

    traj = NamedTrajectory(
        (
            x = 2rand(4, N) .- 1,
            u = u_bound * (2rand(2, N) .- 1),
            du = randn(2, N),
            ddu = randn(2, N),
            Δt = fill(Δt, N),
        );
        controls = (:ddu, :Δt),
        timestep = :Δt,
        bounds = (u = u_bound, Δt = (0.01, 0.5)),
        initial = (x = [1.0, 0.0, 0.0, 0.0], u = zeros(2)),
        final = (u = zeros(2),),
        goal = (x = [0.0, 1.0, 0.0, 0.0],),
    )

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = QuadraticRegularizer(:u, traj, 1.0)
    prob = DirectTrajOptProblem(traj, J, integrators)

    @testset "build_evaluator returns evaluator and Z vector" begin
        evaluator, Z_vec = build_evaluator(prob)
        @test evaluator isa MOI.AbstractNLPEvaluator
        @test length(Z_vec) == traj.dim * traj.N + traj.global_dim

        # Verify eval functions are callable
        obj = MOI.eval_objective(evaluator, Z_vec)
        @test obj isa Float64
        @test isfinite(obj)
    end

    @testset "evaluator_dims returns correct sizes" begin
        evaluator, Z_vec = build_evaluator(prob)
        dims = evaluator_dims(evaluator)
        @test dims.n_constraints == evaluator.n_constraints
        @test dims.n_variables == length(Z_vec)
        @test dims.n_jacobian_entries == length(evaluator.jacobian_structure)
        @test dims.n_hessian_entries == length(evaluator.hessian_structure)
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL — `build_evaluator` not defined

- [ ] **Step 3: Implement build_evaluator and evaluator_dims**

```julia
# src/harness.jl
using DirectTrajOpt
using NamedTrajectories
using MathOptInterface
const MOI = MathOptInterface

"""
    build_evaluator(prob::DirectTrajOptProblem; eval_hessian=true) -> (evaluator, Z_vec)

Extract a MOI evaluator and the initial decision variable vector from a
DirectTrajOptProblem. Used for micro-benchmarking individual eval functions.

Returns:
- `evaluator`: An `MOI.AbstractNLPEvaluator` ready for `MOI.eval_*` calls
- `Z_vec`: The flat decision variable vector `[trajectory_data; global_data]`
"""
function build_evaluator(prob::DirectTrajOpt.Problems.DirectTrajOptProblem; eval_hessian::Bool=true)
    evaluator = DirectTrajOpt.Solvers.Evaluator(prob; eval_hessian=eval_hessian, verbose=false)
    traj = prob.trajectory
    Z_vec = vcat(collect(traj.datavec), collect(traj.global_data))
    return evaluator, Z_vec
end

"""
    evaluator_dims(evaluator) -> NamedTuple

Return key dimensions of the evaluator for buffer pre-allocation.
"""
function evaluator_dims(evaluator::DirectTrajOpt.Solvers.Evaluator)
    return (
        n_constraints = evaluator.n_constraints,
        n_variables = evaluator.trajectory.dim * evaluator.trajectory.N + evaluator.trajectory.global_dim,
        n_jacobian_entries = length(evaluator.jacobian_structure),
        n_hessian_entries = length(evaluator.hessian_structure),
    )
end
```

- [ ] **Step 4: Update module**

```julia
# src/HarmoniqsBenchmarks.jl
module HarmoniqsBenchmarks

export EvalBenchmark, BenchmarkResult, MicroBenchmarkResult
export save_results, load_results, save_micro_results, load_micro_results
export build_evaluator, evaluator_dims

include("schema.jl")
include("storage.jl")
include("harness.jl")

end
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
git add src/harness.jl src/HarmoniqsBenchmarks.jl test/runtests.jl
git commit -m "feat: add build_evaluator and evaluator_dims harness functions"
```

---

## Task 5: Implement benchmark_solve! Harness

**Files:**
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/harness.jl`
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/test/runtests.jl`

- [ ] **Step 1: Add test for benchmark_solve!**

Append inside the `@testset "Harness"` block in `test/runtests.jl`:

```julia
@testset "benchmark_solve! captures metrics" begin
    # Rebuild a fresh problem (solve! mutates in place)
    traj2 = NamedTrajectory(
        (
            x = 2rand(4, N) .- 1,
            u = u_bound * (2rand(2, N) .- 1),
            du = randn(2, N),
            ddu = randn(2, N),
            Δt = fill(Δt, N),
        );
        controls = (:ddu, :Δt),
        timestep = :Δt,
        bounds = (u = u_bound, Δt = (0.01, 0.5)),
        initial = (x = [1.0, 0.0, 0.0, 0.0], u = zeros(2)),
        final = (u = zeros(2),),
        goal = (x = [0.0, 1.0, 0.0, 0.0],),
    )
    integrators2 = [
        BilinearIntegrator(G, :x, :u, traj2),
        DerivativeIntegrator(:u, :du, traj2),
        DerivativeIntegrator(:du, :ddu, traj2),
    ]
    J2 = QuadraticRegularizer(:u, traj2, 1.0)
    prob2 = DirectTrajOptProblem(traj2, J2, integrators2)

    result = benchmark_solve!(
        prob2, IpoptOptions(max_iter=10, print_level=0);
        benchmark_name = "test_solve",
    )

    @test result isa BenchmarkResult
    @test result.package == "DirectTrajOpt"
    @test result.solver == "ipopt"
    @test result.wall_time_s > 0.0
    @test result.iterations >= 0
    @test result.total_allocations_bytes >= 0
    @test result.gc_count >= 0
    @test result.N == N
    @test result.state_dim == 4
    @test haskey(result.solver_options, :max_iter)
    @test result.solver_options[:max_iter] == 10
end
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL — `benchmark_solve!` not defined

- [ ] **Step 3: Implement benchmark_solve!**

Append to `src/harness.jl`:

```julia
using Dates

"""
    benchmark_solve!(prob, options; benchmark_name, runner="local", kwargs...) -> BenchmarkResult

Run `solve!(prob; options, kwargs...)` and capture timing, memory, GC stats, and solver options.
"""
function benchmark_solve!(
    prob::DirectTrajOpt.Problems.DirectTrajOptProblem,
    options::DirectTrajOpt.Solvers.AbstractSolverOptions;
    benchmark_name::String = "unnamed",
    runner::String = "local",
    verbose::Bool = false,
    kwargs...,
)
    traj = prob.trajectory

    # Capture problem dimensions before solve
    n_vars = traj.dim * traj.N + traj.global_dim
    state_dim = _infer_state_dim(prob)
    control_dim = _infer_control_dim(prob)
    n_constraints_total = _count_constraints(prob, options)

    # Snapshot solver options
    opts_snapshot = Dict{Symbol,Any}()
    for name in fieldnames(typeof(options))
        opts_snapshot[name] = getfield(options, name)
    end

    # GC baseline
    GC.gc()
    gc_before = Base.gc_num()

    # Timed solve
    timed = @timed solve!(prob; options=options, verbose=verbose, kwargs...)

    gc_after = Base.gc_num()

    # Compute GC deltas
    gc_time = timed.gctime  # in seconds, convert to ns
    gc_count_delta = gc_after.pause - gc_before.pause
    gc_full_delta = gc_after.full_sweep - gc_before.full_sweep

    # Package version from Project.toml
    pkg_version = _get_package_version("DirectTrajOpt")
    commit = _get_git_commit()

    return BenchmarkResult(
        package = "DirectTrajOpt",
        package_version = pkg_version,
        commit = commit,
        benchmark_name = benchmark_name,
        N = traj.N,
        state_dim = state_dim,
        control_dim = control_dim,
        n_constraints = n_constraints_total,
        n_variables = n_vars,
        wall_time_s = timed.time,
        iterations = -1,  # TODO: extract from solver output when available
        objective_value = NaN,  # TODO: extract from solver
        constraint_violation = NaN,
        solver_status = :Unknown,
        solver = _solver_name(options),
        total_allocations_bytes = timed.bytes,
        total_allocs_count = -1,  # @timed doesn't give count; use gc_num delta
        gc_time_ns = round(Int, timed.gctime * 1e9),
        gc_count = gc_count_delta,
        gc_full_count = gc_full_delta,
        solver_options = opts_snapshot,
        julia_version = string(VERSION),
        timestamp = now(),
        runner = runner,
        n_threads = Threads.nthreads(),
    )
end

# --- helpers ---

function _solver_name(options::DirectTrajOpt.Solvers.AbstractSolverOptions)
    name = string(typeof(options).name.name)
    if occursin("Ipopt", name)
        return "ipopt"
    elseif occursin("MadNLP", name)
        return "madnlp"
    else
        return lowercase(name)
    end
end

function _infer_state_dim(prob)
    traj = prob.trajectory
    # Heuristic: look for common state variable names
    for name in [:x, :ψ̃, :Ũ⃗, :ρ̃]
        if haskey(traj.dims, name)
            return traj.dims[name]
        end
    end
    # Fallback: first non-control component
    return first(values(traj.dims))
end

function _infer_control_dim(prob)
    traj = prob.trajectory
    total = 0
    for name in traj.control_names
        if name != traj.timestep_name
            total += traj.dims[name]
        end
    end
    return total
end

function _count_constraints(prob, options)
    n_dynamics = sum(integrator.dim for integrator in prob.integrators; init=0)
    n_nonlinear = sum(
        c.dim for c in prob.constraints
        if c isa DirectTrajOpt.Constraints.AbstractNonlinearConstraint;
        init=0
    )
    return n_dynamics * (prob.trajectory.N - 1) + n_nonlinear
end

function _get_package_version(pkg_name::String)
    try
        deps = Pkg.dependencies()
        for (_, info) in deps
            if info.name == pkg_name
                return string(info.version)
            end
        end
    catch
    end
    return "unknown"
end

function _get_git_commit()
    try
        return strip(read(`git rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end
```

- [ ] **Step 4: Add `Pkg` import to harness.jl**

Add at the top of `src/harness.jl`:

```julia
import Pkg
```

- [ ] **Step 5: Update module exports**

In `src/HarmoniqsBenchmarks.jl`, add to exports:

```julia
export benchmark_solve!
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
git add src/harness.jl src/HarmoniqsBenchmarks.jl test/runtests.jl
git commit -m "feat: add benchmark_solve! harness with GC stats and options snapshot"
```

---

## Task 6: Implement BenchmarkTools→EvalBenchmark Conversion

**Files:**
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/harness.jl`
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/test/runtests.jl`

- [ ] **Step 1: Add test for trial_to_eval_benchmark**

Append inside `@testset "Harness"`:

```julia
@testset "trial_to_eval_benchmark extracts data from BenchmarkTools.Trial" begin
    using BenchmarkTools
    trial = @benchmark 1 + 1
    eb = trial_to_eval_benchmark(trial)
    @test eb isa EvalBenchmark
    @test length(eb.times_ns) > 0
    @test eb.min_ns > 0.0
    @test eb.memory_bytes >= 0
    @test eb.allocs >= 0
end
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL — `trial_to_eval_benchmark` not defined

- [ ] **Step 3: Implement trial_to_eval_benchmark**

Append to `src/harness.jl`:

```julia
using BenchmarkTools

"""
    trial_to_eval_benchmark(trial::BenchmarkTools.Trial) -> EvalBenchmark

Convert a BenchmarkTools.Trial to an EvalBenchmark, extracting raw timing data.
"""
function trial_to_eval_benchmark(trial::BenchmarkTools.Trial)
    return EvalBenchmark(
        times_ns = Float64.(trial.times),
        gctimes_ns = Float64.(trial.gctimes),
        memory_bytes = trial.memory,
        allocs = trial.allocs,
    )
end
```

- [ ] **Step 4: Export the function**

Add `trial_to_eval_benchmark` to exports in `src/HarmoniqsBenchmarks.jl`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
git add src/harness.jl src/HarmoniqsBenchmarks.jl test/runtests.jl
git commit -m "feat: add trial_to_eval_benchmark for BenchmarkTools integration"
```

---

## Task 7: Implement compare_results Reporter

**Files:**
- Create: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/report.jl`
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/src/HarmoniqsBenchmarks.jl`
- Modify: `/home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl/test/runtests.jl`

- [ ] **Step 1: Add test for compare_results**

Append to `test/runtests.jl`, inside top-level testset:

```julia
@testset "Report" begin
    @testset "compare_results detects regressions" begin
        baseline = BenchmarkResult(
            package="DirectTrajOpt", package_version="0.8.9",
            commit="aaa1111", benchmark_name="test",
            N=51, state_dim=4, control_dim=2,
            n_constraints=200, n_variables=765,
            wall_time_s=1.0, iterations=50,
            objective_value=0.001, constraint_violation=1e-8,
            solver_status=:Optimal, solver="ipopt",
            total_allocations_bytes=1_000_000, total_allocs_count=500,
            gc_time_ns=10_000, gc_count=2, gc_full_count=0,
            solver_options=Dict{Symbol,Any}(),
            julia_version=string(VERSION), timestamp=now(),
            runner="local", n_threads=1,
        )

        # 20% regression in wall time
        current = BenchmarkResult(
            package="DirectTrajOpt", package_version="0.8.10",
            commit="bbb2222", benchmark_name="test",
            N=51, state_dim=4, control_dim=2,
            n_constraints=200, n_variables=765,
            wall_time_s=1.2, iterations=50,
            objective_value=0.001, constraint_violation=1e-8,
            solver_status=:Optimal, solver="ipopt",
            total_allocations_bytes=900_000, total_allocs_count=450,
            gc_time_ns=10_000, gc_count=2, gc_full_count=0,
            solver_options=Dict{Symbol,Any}(),
            julia_version=string(VERSION), timestamp=now(),
            runner="local", n_threads=1,
        )

        comparison = compare_results([baseline], [current])
        @test length(comparison) == 1
        row = comparison[1]
        @test row.benchmark_name == "test"
        @test row.wall_time_pct_change > 15.0  # 20% regression
        @test row.alloc_bytes_pct_change < 0.0  # 10% improvement
        @test row.has_regression == true         # wall time regressed >10%
    end
end
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: FAIL — `compare_results` not defined

- [ ] **Step 3: Implement compare_results**

```julia
# src/report.jl

struct ComparisonRow
    benchmark_name::String
    solver::String
    N::Int
    state_dim::Int
    # Wall time
    baseline_wall_s::Float64
    current_wall_s::Float64
    wall_time_pct_change::Float64
    # Allocations
    baseline_alloc_bytes::Int
    current_alloc_bytes::Int
    alloc_bytes_pct_change::Float64
    # Regression flag
    has_regression::Bool
end

"""
    compare_results(baseline, current; regression_threshold=10.0) -> Vector{ComparisonRow}

Compare two sets of BenchmarkResults by matching on `benchmark_name`.
Returns comparison rows with percent changes and regression flags.

A regression is flagged when wall_time or allocations increase by more than
`regression_threshold` percent.
"""
function compare_results(
    baseline::Vector{BenchmarkResult},
    current::Vector{BenchmarkResult};
    regression_threshold::Float64 = 10.0,
)
    baseline_by_name = Dict(r.benchmark_name => r for r in baseline)
    rows = ComparisonRow[]

    for r in current
        b = get(baseline_by_name, r.benchmark_name, nothing)
        isnothing(b) && continue

        wall_pct = _pct_change(b.wall_time_s, r.wall_time_s)
        alloc_pct = _pct_change(Float64(b.total_allocations_bytes), Float64(r.total_allocations_bytes))
        has_regression = wall_pct > regression_threshold || alloc_pct > regression_threshold

        push!(rows, ComparisonRow(
            r.benchmark_name, r.solver, r.N, r.state_dim,
            b.wall_time_s, r.wall_time_s, wall_pct,
            b.total_allocations_bytes, r.total_allocations_bytes, alloc_pct,
            has_regression,
        ))
    end

    return rows
end

function _pct_change(old::Float64, new::Float64)
    old == 0.0 && return new == 0.0 ? 0.0 : 100.0
    return (new - old) / abs(old) * 100.0
end
```

- [ ] **Step 4: Update module**

Add exports to `src/HarmoniqsBenchmarks.jl`:

```julia
export compare_results, ComparisonRow
```

And add the include:

```julia
include("report.jl")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /home/jack/repos/harmoniqs/HarmoniqsBenchmarks.jl
git add src/report.jl src/HarmoniqsBenchmarks.jl test/runtests.jl
git commit -m "feat: add compare_results reporter with regression detection"
```

---

## Task 8: Set Up DirectTrajOpt.jl Benchmark Environment

**Files:**
- Create: `/home/jack/repos/harmoniqs/DirectTrajOpt.jl/benchmark/Project.toml`
- Create: `/home/jack/repos/harmoniqs/DirectTrajOpt.jl/benchmark/.gitignore`
- Create: `/home/jack/repos/harmoniqs/DirectTrajOpt.jl/benchmark/benchmarks.jl`

- [ ] **Step 1: Create benchmark directory**

```bash
mkdir -p /home/jack/repos/harmoniqs/DirectTrajOpt.jl/benchmark/results
```

- [ ] **Step 2: Create .gitignore**

```
# benchmark/.gitignore
results/
```

- [ ] **Step 3: Create benchmark/Project.toml**

```toml
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DirectTrajOpt = "c823fa1f-8872-4af5-b810-2b9b72bbbf56"
ExponentialAction = "e24c0720-ea99-47e8-929e-571b494574d3"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HarmoniqsBenchmarks = "INSERT_UUID"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MadNLP = "2621e9c9-9eb4-46b1-8089-e8c72242dfb6"
MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
NamedTrajectories = "538bc3a1-5ab9-4fc3-b776-35ca1e893e08"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
TestItemRunner = "f8b46487-2199-4994-9208-9a1283c18c0a"
TestItems = "1c621080-faea-4a02-84b6-bbd5e436b8fe"
```

Replace `INSERT_UUID` with the UUID generated in Task 1.

- [ ] **Step 4: Instantiate the benchmark environment**

```bash
cd /home/jack/repos/harmoniqs/DirectTrajOpt.jl
julia --project=benchmark -e '
    using Pkg
    Pkg.develop(path=".")
    Pkg.develop(path="../HarmoniqsBenchmarks.jl")
    Pkg.develop(path="../NamedTrajectories.jl")
    Pkg.instantiate()
    using HarmoniqsBenchmarks
    println("Benchmark env OK")
'
```

Expected: "Benchmark env OK"

- [ ] **Step 5: Create benchmarks.jl stub**

```julia
# benchmark/benchmarks.jl
using TestItems
```

- [ ] **Step 6: Commit**

```bash
cd /home/jack/repos/harmoniqs/DirectTrajOpt.jl
git add benchmark/Project.toml benchmark/.gitignore benchmark/benchmarks.jl
git commit -m "feat: add benchmark/ environment for HarmoniqsBenchmarks integration"
```

---

## Task 9: Write Evaluator Micro-benchmarks

**Files:**
- Modify: `/home/jack/repos/harmoniqs/DirectTrajOpt.jl/benchmark/benchmarks.jl`

- [ ] **Step 1: Write the micro-benchmark @testitem**

```julia
# benchmark/benchmarks.jl
using TestItems

@testitem "Evaluator micro-benchmarks: bilinear N=51" begin
    using HarmoniqsBenchmarks
    using BenchmarkTools
    using DirectTrajOpt
    using NamedTrajectories
    using SparseArrays
    using ExponentialAction
    using MathOptInterface
    const MOI = MathOptInterface
    using Dates

    # Build a deterministic bilinear problem
    Random.seed!(42)
    N = 51; Δt = 0.1; u_bound = 0.1; ω = 0.1
    Gx = sparse(Float64[0 0 0 1; 0 0 1 0; 0 -1 0 0; -1 0 0 0])
    Gy = sparse(Float64[0 -1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 1 0])
    Gz = sparse(Float64[0 0 1 0; 0 0 0 -1; -1 0 0 0; 0 1 0 0])
    G(u) = ω * Gz + u[1] * Gx + u[2] * Gy

    traj = NamedTrajectory(
        (
            x = 2rand(4, N) .- 1,
            u = u_bound * (2rand(2, N) .- 1),
            du = randn(2, N),
            ddu = randn(2, N),
            Δt = fill(Δt, N),
        );
        controls = (:ddu, :Δt),
        timestep = :Δt,
        bounds = (u = u_bound, Δt = (0.01, 0.5)),
        initial = (x = [1.0, 0.0, 0.0, 0.0], u = zeros(2)),
        final = (u = zeros(2),),
        goal = (x = [0.0, 1.0, 0.0, 0.0],),
    )

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = QuadraticRegularizer(:u, traj, 1.0) + QuadraticRegularizer(:du, traj, 1.0)
    prob = DirectTrajOptProblem(traj, J, integrators)

    evaluator, Z_vec = build_evaluator(prob)
    dims = evaluator_dims(evaluator)

    # Pre-allocate buffers
    g = zeros(dims.n_constraints)
    grad = zeros(dims.n_variables)
    H = zeros(dims.n_hessian_entries)
    Jac = zeros(dims.n_jacobian_entries)
    sigma = 1.0
    mu = ones(dims.n_constraints)

    # Run benchmarks
    benchmarks = Dict{Symbol,EvalBenchmark}(
        :eval_objective => trial_to_eval_benchmark(
            @benchmark(MOI.eval_objective($evaluator, $Z_vec))
        ),
        :eval_gradient => trial_to_eval_benchmark(
            @benchmark(MOI.eval_objective_gradient($evaluator, $grad, $Z_vec))
        ),
        :eval_constraint => trial_to_eval_benchmark(
            @benchmark(MOI.eval_constraint($evaluator, $g, $Z_vec))
        ),
        :eval_jacobian => trial_to_eval_benchmark(
            @benchmark(MOI.eval_constraint_jacobian($evaluator, $Jac, $Z_vec))
        ),
        :eval_hessian_lagrangian => trial_to_eval_benchmark(
            @benchmark(MOI.eval_hessian_lagrangian($evaluator, $H, $Z_vec, $sigma, $mu))
        ),
    )

    result = MicroBenchmarkResult(
        package = "DirectTrajOpt",
        package_version = "0.8.10",
        commit = try strip(read(`git rev-parse --short HEAD`, String)) catch; "unknown" end,
        benchmark_name = "evaluator_micro_bilinear_N51",
        N = N, state_dim = 4, control_dim = 2,
        eval_benchmarks = benchmarks,
        julia_version = string(VERSION),
        timestamp = now(),
        runner = get(ENV, "BENCHMARK_RUNNER", "local"),
        n_threads = Threads.nthreads(),
    )

    # Print summary
    println("\n=== Evaluator Micro-benchmarks (bilinear N=$N) ===")
    for (name, eb) in sort(collect(result.eval_benchmarks), by=first)
        Printf = Base.Printf
        @Printf.printf("  %-25s  median: %8.1f ns  allocs: %d  memory: %d bytes\n",
            name, eb.median_ns, eb.allocs, eb.memory_bytes)
    end

    # Save
    results_dir = joinpath(@__DIR__, "results")
    save_micro_results(results_dir, result.benchmark_name, result)
    println("  Saved to $results_dir/")
end
```

- [ ] **Step 2: Run the micro-benchmark to verify it works**

```bash
cd /home/jack/repos/harmoniqs/DirectTrajOpt.jl
julia --project=benchmark -e '
    using TestItemRunner
    @run_package_tests(filter=ti -> occursin("micro", ti.name), benchmark)
'
```

Expected: Benchmark runs, prints timing table, saves JLD2 to `benchmark/results/`

- [ ] **Step 3: Verify the JLD2 output is loadable**

```bash
cd /home/jack/repos/harmoniqs/DirectTrajOpt.jl
julia --project=benchmark -e '
    using HarmoniqsBenchmarks
    files = filter(f -> endswith(f, ".jld2"), readdir("benchmark/results", join=true))
    @assert length(files) >= 1 "Expected at least one JLD2 file"
    result = load_micro_results(files[1])
    println("Loaded: $(result.benchmark_name)")
    println("Functions benchmarked: $(keys(result.eval_benchmarks))")
'
```

Expected: Loads successfully, shows function names

- [ ] **Step 4: Commit**

```bash
cd /home/jack/repos/harmoniqs/DirectTrajOpt.jl
git add benchmark/benchmarks.jl
git commit -m "feat: add evaluator micro-benchmarks with BenchmarkTools"
```

---

## Task 10: Write Ipopt vs MadNLP Macro-benchmarks

**Files:**
- Modify: `/home/jack/repos/harmoniqs/DirectTrajOpt.jl/benchmark/benchmarks.jl`

- [ ] **Step 1: Append the macro-benchmark @testitem**

Append to `benchmark/benchmarks.jl`:

```julia
@testitem "Ipopt vs MadNLP: bilinear N=51" begin
    using HarmoniqsBenchmarks
    using DirectTrajOpt
    using NamedTrajectories
    using SparseArrays
    using ExponentialAction
    import MadNLP
    using Dates

    # Resolve MadNLPOptions from the extension
    const MadNLPSolverExt = [
        mod for mod in reverse(Base.loaded_modules_order)
        if Symbol(mod) == :MadNLPSolverExt
    ][1]

    function make_bilinear_problem(; seed=42)
        Random.seed!(seed)
        N = 51; Δt = 0.1; u_bound = 0.1; ω = 0.1
        Gx = sparse(Float64[0 0 0 1; 0 0 1 0; 0 -1 0 0; -1 0 0 0])
        Gy = sparse(Float64[0 -1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 1 0])
        Gz = sparse(Float64[0 0 1 0; 0 0 0 -1; -1 0 0 0; 0 1 0 0])
        G(u) = ω * Gz + u[1] * Gx + u[2] * Gy

        traj = NamedTrajectory(
            (
                x = 2rand(4, N) .- 1,
                u = u_bound * (2rand(2, N) .- 1),
                du = randn(2, N),
                ddu = randn(2, N),
                Δt = fill(Δt, N),
            );
            controls = (:ddu, :Δt),
            timestep = :Δt,
            bounds = (u = u_bound, Δt = (0.01, 0.5)),
            initial = (x = [1.0, 0.0, 0.0, 0.0], u = zeros(2)),
            final = (u = zeros(2),),
            goal = (x = [0.0, 1.0, 0.0, 0.0],),
        )

        integrators = [
            BilinearIntegrator(G, :x, :u, traj),
            DerivativeIntegrator(:u, :du, traj),
            DerivativeIntegrator(:du, :ddu, traj),
        ]
        J = QuadraticRegularizer(:u, traj, 1.0) + QuadraticRegularizer(:du, traj, 1.0)
        return DirectTrajOptProblem(traj, J, integrators)
    end

    # Ipopt solve
    prob_ipopt = make_bilinear_problem()
    result_ipopt = benchmark_solve!(
        prob_ipopt,
        IpoptOptions(max_iter=200, print_level=0);
        benchmark_name = "bilinear_N51_ipopt",
    )

    # MadNLP solve (fresh problem)
    prob_madnlp = make_bilinear_problem()
    result_madnlp = benchmark_solve!(
        prob_madnlp,
        MadNLPSolverExt.MadNLPOptions(max_iter=200, print_level=1);
        benchmark_name = "bilinear_N51_madnlp",
    )

    # Print comparison
    println("\n=== Ipopt vs MadNLP: bilinear N=51 ===")
    println("  Ipopt:  $(round(result_ipopt.wall_time_s, digits=3))s, $(result_ipopt.total_allocations_bytes ÷ 1024) KB alloc")
    println("  MadNLP: $(round(result_madnlp.wall_time_s, digits=3))s, $(result_madnlp.total_allocations_bytes ÷ 1024) KB alloc")

    # Save
    results_dir = joinpath(@__DIR__, "results")
    save_results(results_dir, "ipopt_vs_madnlp_N51", [result_ipopt, result_madnlp])
end
```

- [ ] **Step 2: Run the macro-benchmark**

```bash
cd /home/jack/repos/harmoniqs/DirectTrajOpt.jl
julia --project=benchmark -e '
    using TestItemRunner
    @run_package_tests(filter=ti -> occursin("Ipopt vs MadNLP", ti.name), benchmark)
'
```

Expected: Both solvers run, prints wall time and allocation comparison

- [ ] **Step 3: Commit**

```bash
cd /home/jack/repos/harmoniqs/DirectTrajOpt.jl
git add benchmark/benchmarks.jl
git commit -m "feat: add Ipopt vs MadNLP macro-benchmark"
```

---

## Task 11: Write Memory Scaling Study

**Files:**
- Modify: `/home/jack/repos/harmoniqs/DirectTrajOpt.jl/benchmark/benchmarks.jl`

- [ ] **Step 1: Append the scaling study @testitem**

Append to `benchmark/benchmarks.jl`:

```julia
@testitem "Memory scaling: N and state_dim sweep" begin
    using HarmoniqsBenchmarks
    using DirectTrajOpt
    using NamedTrajectories
    using SparseArrays
    using ExponentialAction
    import MadNLP
    using Dates, Printf

    const MadNLPSolverExt = [
        mod for mod in reverse(Base.loaded_modules_order)
        if Symbol(mod) == :MadNLPSolverExt
    ][1]

    function make_scaled_problem(; N, state_dim, n_controls=2, seed=42)
        Random.seed!(seed)

        # Build random bilinear system at given state dimension
        G_drift = sparse(randn(state_dim, state_dim))
        G_drives = [sparse(randn(state_dim, state_dim)) for _ in 1:n_controls]
        G(u) = G_drift + sum(u[i] * G_drives[i] for i in 1:n_controls)

        x_init = zeros(state_dim); x_init[1] = 1.0
        x_goal = zeros(state_dim); x_goal[2] = 1.0

        traj = NamedTrajectory(
            (
                x = randn(state_dim, N),
                u = 0.1 * randn(n_controls, N),
                du = randn(n_controls, N),
                Δt = fill(0.1, N),
            );
            controls = (:du, :Δt),
            timestep = :Δt,
            bounds = (u = 1.0, Δt = (0.01, 0.5)),
            initial = (x = x_init, u = zeros(n_controls)),
            final = (u = zeros(n_controls),),
            goal = (x = x_goal,),
        )

        integrators = [
            BilinearIntegrator(G, :x, :u, traj),
            DerivativeIntegrator(:u, :du, traj),
        ]
        J = QuadraticRegularizer(:u, traj, 1.0)
        return DirectTrajOptProblem(traj, J, integrators)
    end

    N_values = [25, 51, 101]
    dim_values = [4, 8, 16]
    results = BenchmarkResult[]

    println("\n=== Memory Scaling Study ===")
    @printf("  %5s | %5s | %12s | %12s | %12s | %12s\n",
        "N", "dim", "Ipopt (s)", "Ipopt (KB)", "MadNLP (s)", "MadNLP (KB)")
    @printf("  %5s-+-%5s-+-%12s-+-%12s-+-%12s-+-%12s\n",
        "-"^5, "-"^5, "-"^12, "-"^12, "-"^12, "-"^12)

    for N in N_values
        for dim in dim_values
            # Ipopt
            prob = make_scaled_problem(; N=N, state_dim=dim)
            r_ipopt = benchmark_solve!(
                prob, IpoptOptions(max_iter=50, print_level=0);
                benchmark_name = "scaling_N$(N)_d$(dim)_ipopt",
            )
            push!(results, r_ipopt)

            # MadNLP
            prob = make_scaled_problem(; N=N, state_dim=dim)
            r_madnlp = benchmark_solve!(
                prob, MadNLPSolverExt.MadNLPOptions(max_iter=50, print_level=1);
                benchmark_name = "scaling_N$(N)_d$(dim)_madnlp",
            )
            push!(results, r_madnlp)

            @printf("  %5d | %5d | %12.3f | %12d | %12.3f | %12d\n",
                N, dim,
                r_ipopt.wall_time_s, r_ipopt.total_allocations_bytes ÷ 1024,
                r_madnlp.wall_time_s, r_madnlp.total_allocations_bytes ÷ 1024)
        end
    end

    # Save all results
    results_dir = joinpath(@__DIR__, "results")
    save_results(results_dir, "memory_scaling", results)
    println("\n  Saved $(length(results)) results to $results_dir/")
end
```

- [ ] **Step 2: Run the scaling study**

```bash
cd /home/jack/repos/harmoniqs/DirectTrajOpt.jl
julia --project=benchmark -e '
    using TestItemRunner
    @run_package_tests(filter=ti -> occursin("Memory scaling", ti.name), benchmark)
'
```

Expected: Table printed with wall times and allocations for each (N, dim) combination

- [ ] **Step 3: Commit**

```bash
cd /home/jack/repos/harmoniqs/DirectTrajOpt.jl
git add benchmark/benchmarks.jl
git commit -m "feat: add memory scaling study benchmark (N x state_dim sweep)"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] `cd HarmoniqsBenchmarks.jl && julia --project=. -e 'using Pkg; Pkg.test()'` — all tests pass
- [ ] `cd DirectTrajOpt.jl && julia --project=benchmark -e 'using TestItemRunner; @run_package_tests(benchmark)'` — all three benchmark @testitems run
- [ ] `ls DirectTrajOpt.jl/benchmark/results/` — contains `.jld2` files for each benchmark
- [ ] Load and compare results:
  ```julia
  using HarmoniqsBenchmarks
  results = load_results("benchmark/results/ipopt_vs_madnlp_N51_<sha>.jld2")
  println("Ipopt: $(results[1].wall_time_s)s, MadNLP: $(results[2].wall_time_s)s")
  ```

---

## Follow-up Plans (Not in Scope)

- **Piccolissimo benchmark suite** — migrate existing `benchmark/complex_vs_real_ode.jl` and `constraint_comparison.jl` to use HarmoniqsBenchmarks schema
- **Demo-repo problem generators** — clone bosonic-demo, nv-center-demo, atoms-demo, ions, fluxonium-demo, gkp-stanford and extract system Hamiltonians
- **CI workflows** — `.github/workflows/benchmark.yml` for DirectTrajOpt and other packages
- **Allocation profiling spike** — parallel worktree experiments with Profile.Allocs, AllocCheck.jl, --track-allocation
- **Aggregator repo** — `harmoniqs-benchmarks` with cross-package comparison tables
