using TestItems

@testitem "Evaluator micro-benchmarks: bilinear N=51" begin
    using HarmoniqsBenchmarks, BenchmarkTools, DirectTrajOpt, NamedTrajectories
    using SparseArrays, ExponentialAction, MathOptInterface, Random, Dates, Printf
    const MOI = MathOptInterface

    Random.seed!(42)
    N = 51;
    Δt = 0.1;
    u_bound = 0.1;
    ω = 0.1
    Gx = sparse(Float64[0 0 0 1; 0 0 1 0; 0 -1 0 0; -1 0 0 0])
    Gy = sparse(Float64[0 -1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 1 0])
    Gz = sparse(Float64[0 0 1 0; 0 0 0 -1; -1 0 0 0; 0 1 0 0])
    G(u) = ω * Gz + u[1] * Gx + u[2] * Gy

    traj = NamedTrajectory(
        (
            x = 2rand(4, N) .- 1,
            u = u_bound*(2rand(2, N) .- 1),
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

    result = MicroBenchmarkResult(
        package = "DirectTrajOpt",
        package_version = "0.8.10",
        commit = (
            try
                String(strip(read(`git rev-parse --short HEAD`, String)))
            catch
                ; "unknown"
            end
        ),
        benchmark_name = "evaluator_micro_bilinear_N51",
        N = N,
        state_dim = 4,
        control_dim = 2,
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

    const MadNLPSolverExt = [
        mod for mod in reverse(Base.loaded_modules_order) if Symbol(mod) == :MadNLPSolverExt
    ][1]

    function make_bilinear_problem(; seed = 42)
        Random.seed!(seed)
        N = 51;
        Δt = 0.1;
        u_bound = 0.1;
        ω = 0.1
        Gx = sparse(Float64[0 0 0 1; 0 0 1 0; 0 -1 0 0; -1 0 0 0])
        Gy = sparse(Float64[0 -1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 1 0])
        Gz = sparse(Float64[0 0 1 0; 0 0 0 -1; -1 0 0 0; 0 1 0 0])
        G(u) = ω * Gz + u[1] * Gx + u[2] * Gy

        traj = NamedTrajectory(
            (
                x = 2rand(4, N) .- 1,
                u = u_bound*(2rand(2, N) .- 1),
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

    prob_ipopt = make_bilinear_problem()
    result_ipopt = benchmark_solve!(
        prob_ipopt,
        IpoptOptions(max_iter = 200, print_level = 0);
        benchmark_name = "bilinear_N51_ipopt",
    )

    prob_madnlp = make_bilinear_problem()
    result_madnlp = benchmark_solve!(
        prob_madnlp,
        MadNLPSolverExt.MadNLPOptions(max_iter = 200, print_level = 1);
        benchmark_name = "bilinear_N51_madnlp",
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

    const MadNLPSolverExt = [
        mod for mod in reverse(Base.loaded_modules_order) if Symbol(mod) == :MadNLPSolverExt
    ][1]

    function make_scaled_problem(; N, state_dim, n_controls = 2, seed = 42)
        Random.seed!(seed)
        G_drift = sparse(randn(state_dim, state_dim))
        G_drives = [sparse(randn(state_dim, state_dim)) for _ = 1:n_controls]
        G(u) = G_drift + sum(u[i] * G_drives[i] for i = 1:n_controls)

        x_init = zeros(state_dim);
        x_init[1] = 1.0
        x_goal = zeros(state_dim);
        x_goal[min(2, state_dim)] = 1.0

        traj = NamedTrajectory(
            (
                x = randn(state_dim, N),
                u = 0.1*randn(n_controls, N),
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
        integrators =
            [BilinearIntegrator(G, :x, :u, traj), DerivativeIntegrator(:u, :du, traj)]
        J = QuadraticRegularizer(:u, traj, 1.0)
        return DirectTrajOptProblem(traj, J, integrators)
    end

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
            )
            push!(results, r_ipopt)

            prob = make_scaled_problem(; N = N, state_dim = dim)
            r_madnlp = benchmark_solve!(
                prob,
                MadNLPSolverExt.MadNLPOptions(max_iter = 50, print_level = 1);
                benchmark_name = "scaling_N$(N)_d$(dim)_madnlp",
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
