using TestItems

@testitem "Alloc profile: bilinear N=51 (Ipopt + MadNLP)" begin
    using HarmoniqsBenchmarks, DirectTrajOpt, NamedTrajectories
    using SparseArrays, ExponentialAction, Random, Dates
    import MadNLP

    include("$(joinpath(@__DIR__, "problem_utils.jl"))")

    runner = get(ENV, "BENCHMARK_RUNNER", "local")

    # `Profile.Allocs` slows the solve dramatically — `sample_rate = 1.0` is
    # intractable for a full Ipopt/MadNLP run (>15 min on N=10 in early
    # experiments), and even `0.01` runs MadNLP at ~3000× slowdown vs the
    # un-profiled solve. `0.01` keeps the trace tractable while still giving
    # statistically useful per-frame breakdowns; combined with `max_iter = 30`
    # (representative per-iter allocation pattern — convergence isn't the
    # goal) the testitem completes well inside the workflow timeout. The
    # `1 / sample_rate` scaling applied by `report_alloc_profile` extrapolates
    # back to total bytes.
    sample_rate = 0.01

    # JIT warmup so first-call compile of Ipopt/MadNLP extensions, KKT/AD
    # codegen, and the Profile.Allocs machinery itself doesn't dominate the
    # sampled trace. Discard the warmup results.
    let warmup_prob = make_bilinear_problem(; N = 11, seed = 0)
        DirectTrajOpt.solve!(
            warmup_prob;
            options = IpoptOptions(max_iter = 2, print_level = 0),
        )
    end
    let warmup_prob = make_bilinear_problem(; N = 11, seed = 0)
        DirectTrajOpt.solve!(
            warmup_prob;
            options = MadNLPOptions(max_iter = 2, print_level = 6),
        )
    end

    results_dir = joinpath(@__DIR__, "results", "allocs")
    pdims = problem_dims(make_bilinear_problem(; N = 51, seed = 42))

    # Ipopt
    let prob = make_bilinear_problem(; N = 51, seed = 42)
        profile = benchmark_memory!(
            () -> DirectTrajOpt.solve!(
                prob;
                options = IpoptOptions(max_iter = 30, print_level = 0),
            );
            package = "DirectTrajOpt",
            solver = "Ipopt",
            benchmark_name = "alloc_bilinear_N51_ipopt",
            N = 51,
            state_dim = pdims.state_dim,
            control_dim = pdims.control_dim,
            sample_rate = sample_rate,
            warmup = false,  # we did our own warmup above
            runner = runner,
        )
        path = save_alloc_profile(results_dir, profile.benchmark_name, profile)
        println("\n=== Alloc profile: Ipopt (bilinear N=51, sample_rate=$sample_rate) ===")
        println("  samples=$(profile.total_count)  total≈$(profile.total_bytes) B")
        println("  saved $path")
        report_alloc_profile(profile; k_types = 10, k_leaves = 15, k_frames = 15)
    end

    # MadNLP
    let prob = make_bilinear_problem(; N = 51, seed = 42)
        profile = benchmark_memory!(
            () -> DirectTrajOpt.solve!(
                prob;
                options = MadNLPOptions(max_iter = 30, print_level = 6),
            );
            package = "DirectTrajOpt",
            solver = "MadNLP",
            benchmark_name = "alloc_bilinear_N51_madnlp",
            N = 51,
            state_dim = pdims.state_dim,
            control_dim = pdims.control_dim,
            sample_rate = sample_rate,
            warmup = false,
            runner = runner,
        )
        path = save_alloc_profile(results_dir, profile.benchmark_name, profile)
        println("\n=== Alloc profile: MadNLP (bilinear N=51, sample_rate=$sample_rate) ===")
        println("  samples=$(profile.total_count)  total≈$(profile.total_bytes) B")
        println("  saved $path")
        report_alloc_profile(profile; k_types = 10, k_leaves = 15, k_frames = 15)
    end
end
