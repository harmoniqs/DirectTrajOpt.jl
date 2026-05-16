using TestItems

@testitem "X gate convergence: Ipopt" begin
    using HarmoniqsBenchmarks
    using DirectTrajOpt
    using NamedTrajectories
    using SparseArrays, ExponentialAction, Random, Dates, Printf, LinearAlgebra

    include(joinpath(@__DIR__, "problem_utils.jl"))

    runner = get(ENV, "BENCHMARK_RUNNER", "local")

    prob = _make_xgate_prob(; N = 51, seed = 42)

    # Wire Ipopt callback that captures final iter_count + inf_pr.
    state, cb = ipopt_capture()
    ipopt_opts = IpoptOptions(max_iter = 500, print_level = 0)

    # benchmark_solve! forwards extra kwargs to DirectTrajOpt.solve!, so we
    # can inject the capture callback through the same call.
    result = benchmark_solve!(
        prob,
        ipopt_opts;
        benchmark_name = "xgate_convergence_ipopt_N51",
        runner         = runner,
        callback       = cb,
    )

    final_inf  = _xgate_infidelity(prob)
    primal_inf = ipopt_primal_infeasibility(state)
    iters      = ipopt_iterations(state)

    crit = InfidelityConvergence(
        target_infidelity    = 1e-3,
        final_infidelity     = final_inf,
        primal_infeasibility = primal_inf,
        feas_tol             = 1e-6,
    )

    result_with_conv = _build_convergence_result(result, crit; iterations = iters)

    @printf(
        "\n=== X gate convergence (Ipopt) ===\n  iters=%d  final_inf=%.3e  inf_pr=%.3e  wall=%.3fs  converged=%s\n",
        iters,
        final_inf,
        primal_inf,
        result_with_conv.wall_time_s,
        converged(crit),
    )

    @test converged(result_with_conv.convergence) == true

    results_dir = joinpath(@__DIR__, "results")
    saved = save_results(results_dir, "xgate_convergence_ipopt_N51", [result_with_conv])
    println("  Saved $(saved)")

    # Exercise the reporting path.
    rows = compare_convergence([result_with_conv])
    @test length(rows) == 1
    @test rows[1].converged == true
end


@testitem "X gate convergence: MadNLP" begin
    using HarmoniqsBenchmarks
    using DirectTrajOpt
    using NamedTrajectories
    using SparseArrays, ExponentialAction, Random, Dates, Printf, LinearAlgebra
    import MadNLP

    include(joinpath(@__DIR__, "problem_utils.jl"))

    runner = get(ENV, "BENCHMARK_RUNNER", "local")

    prob = _make_xgate_prob(; N = 51, seed = 42)

    madnlp_opts = MadNLPOptions(max_iter = 500, print_level = 6)

    # MadNLP doesn't have an ipopt_capture analogue yet — use the post-solve
    # constraint_violation that benchmark_solve! already extracted as the
    # primal-infeasibility proxy.
    result = benchmark_solve!(
        prob,
        madnlp_opts;
        benchmark_name = "xgate_convergence_madnlp_N51",
        runner         = runner,
    )

    final_inf  = _xgate_infidelity(prob)
    primal_inf = result.constraint_violation

    crit = InfidelityConvergence(
        target_infidelity    = 1e-3,
        final_infidelity     = final_inf,
        primal_infeasibility = primal_inf,
        feas_tol             = 1e-6,
    )

    result_with_conv = _build_convergence_result(result, crit)

    @printf(
        "\n=== X gate convergence (MadNLP) ===\n  final_inf=%.3e  cviol=%.3e  wall=%.3fs  converged=%s\n",
        final_inf,
        primal_inf,
        result_with_conv.wall_time_s,
        converged(crit),
    )

    @test converged(result_with_conv.convergence) == true

    results_dir = joinpath(@__DIR__, "results")
    saved = save_results(results_dir, "xgate_convergence_madnlp_N51", [result_with_conv])
    println("  Saved $(saved)")

    rows = compare_convergence([result_with_conv])
    @test length(rows) == 1
    @test rows[1].converged == true
end
