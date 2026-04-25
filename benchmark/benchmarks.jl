using TestItems

@testitem "Evaluator micro-benchmarks: bilinear N=51" begin
    using HarmoniqsBenchmarks, BenchmarkTools, DirectTrajOpt, NamedTrajectories
    using SparseArrays, ExponentialAction, MathOptInterface, Random, Dates, Printf, Pkg
    const MOI = MathOptInterface

    include("$(joinpath(@__DIR__, "problem_utils.jl"))")

    N = 51
    prob = make_bilinear_problem(; N = N, seed = 42)

    evaluator, Z_vec = build_evaluator(prob)
    dims = evaluator_dims(evaluator)

    g = zeros(dims.n_constraints)
    grad = zeros(dims.n_variables)
    H = zeros(dims.n_hessian_entries)
    Jac = zeros(dims.n_jacobian_entries)
    sigma = 1.0
    mu = ones(dims.n_constraints)

    benchmarks = Dict{Symbol,EvalBenchmark}(
        :eval_objective =>
            trial_to_eval_benchmark(@benchmark(MOI.eval_objective($evaluator, $Z_vec))),
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

    pkg_version = let v = nothing
        try
            for (_, info) in Pkg.dependencies()
                if info.name == "DirectTrajOpt"
                    v = info.version
                    break
                end
            end
        catch
        end
        isnothing(v) ? "unknown" : string(v)
    end

    pdims = problem_dims(prob)

    result = MicroBenchmarkResult(
        package = "DirectTrajOpt",
        package_version = pkg_version,
        commit = (
            try
                String(strip(read(`git rev-parse --short HEAD`, String)))
            catch
                ;
                "unknown"
            end
        ),
        benchmark_name = "evaluator_micro_bilinear_N51",
        N = N,
        state_dim = pdims.state_dim,
        control_dim = pdims.control_dim,
        eval_benchmarks = benchmarks,
        julia_version = string(VERSION),
        timestamp = Dates.now(),
        runner = get(ENV, "BENCHMARK_RUNNER", "local"),
        n_threads = Threads.nthreads(),
    )

    println("\n=== Evaluator Micro-benchmarks (bilinear N=$N) ===")
    for (name, eb) in sort(collect(result.eval_benchmarks), by = first)
        @printf(
            "  %-25s  median: %8.1f ns  allocs: %d  memory: %d bytes\n",
            name,
            eb.median_ns,
            eb.allocs,
            eb.memory_bytes
        )
    end

    results_dir = joinpath(@__DIR__, "results")
    save_micro_results(results_dir, result.benchmark_name, result)
    println("  Saved to $results_dir/")
end

@testitem "Ipopt vs MadNLP: bilinear N=51" begin
    using HarmoniqsBenchmarks, DirectTrajOpt, NamedTrajectories
    using SparseArrays, ExponentialAction, Random, Dates
    import MadNLP

    include("$(joinpath(@__DIR__, "problem_utils.jl"))")

    runner = get(ENV, "BENCHMARK_RUNNER", "local")

    prob_ipopt = make_bilinear_problem(; N = 51, seed = 42)
    result_ipopt = benchmark_solve!(
        prob_ipopt,
        IpoptOptions(max_iter = 200, print_level = 0);
        benchmark_name = "bilinear_N51_ipopt",
        runner = runner,
    )

    prob_madnlp = make_bilinear_problem(; N = 51, seed = 42)
    result_madnlp = benchmark_solve!(
        prob_madnlp,
        MadNLPOptions(max_iter = 200, print_level = 6);
        benchmark_name = "bilinear_N51_madnlp",
        runner = runner,
    )

    println("\n=== Ipopt vs MadNLP: bilinear N=51 ===")
    println(
        "  Ipopt:  $(round(result_ipopt.wall_time_s, digits=3))s, $(result_ipopt.total_allocations_bytes ÷ 1024) KB alloc",
    )
    println(
        "  MadNLP: $(round(result_madnlp.wall_time_s, digits=3))s, $(result_madnlp.total_allocations_bytes ÷ 1024) KB alloc",
    )

    results_dir = joinpath(@__DIR__, "results")
    save_results(results_dir, "ipopt_vs_madnlp_N51", [result_ipopt, result_madnlp])
end

@testitem "Memory scaling: N and state_dim sweep" begin
    using HarmoniqsBenchmarks, DirectTrajOpt, NamedTrajectories
    using SparseArrays, ExponentialAction, Random, Dates, Printf
    import MadNLP

    include("$(joinpath(@__DIR__, "problem_utils.jl"))")

    runner = get(ENV, "BENCHMARK_RUNNER", "local")
    N_values = [25, 51, 101]
    dim_values = [4, 8, 16]
    results = BenchmarkResult[]

    println("\n=== Memory Scaling Study ===")
    @printf(
        "  %5s | %5s | %12s | %12s | %12s | %12s\n",
        "N",
        "dim",
        "Ipopt (s)",
        "Ipopt (KB)",
        "MadNLP (s)",
        "MadNLP (KB)"
    )
    @printf(
        "  %5s-+-%5s-+-%12s-+-%12s-+-%12s-+-%12s\n",
        "-"^5,
        "-"^5,
        "-"^12,
        "-"^12,
        "-"^12,
        "-"^12
    )

    for N in N_values
        for dim in dim_values
            prob = make_scaled_problem(; N = N, state_dim = dim)
            r_ipopt = benchmark_solve!(
                prob,
                IpoptOptions(max_iter = 50, print_level = 0);
                benchmark_name = "scaling_N$(N)_d$(dim)_ipopt",
                runner = runner,
            )
            push!(results, r_ipopt)

            prob = make_scaled_problem(; N = N, state_dim = dim)
            r_madnlp = benchmark_solve!(
                prob,
                MadNLPOptions(max_iter = 50, print_level = 6);
                benchmark_name = "scaling_N$(N)_d$(dim)_madnlp",
                runner = runner,
            )
            push!(results, r_madnlp)

            @printf(
                "  %5d | %5d | %12.3f | %12d | %12.3f | %12d\n",
                N,
                dim,
                r_ipopt.wall_time_s,
                r_ipopt.total_allocations_bytes ÷ 1024,
                r_madnlp.wall_time_s,
                r_madnlp.total_allocations_bytes ÷ 1024
            )
        end
    end

    results_dir = joinpath(@__DIR__, "results")
    save_results(results_dir, "memory_scaling", results)
    println("\n  Saved $(length(results)) results to $results_dir/")
end
