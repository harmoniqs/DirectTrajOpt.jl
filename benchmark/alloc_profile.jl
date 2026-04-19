# =============================================================================
# Ipopt + MadNLP allocation profile — bilinear toy problem
#
# Runs `solve!` once per solver under Profile.Allocs via benchmark_memory!
# from HarmoniqsBenchmarks.jl and saves the sampled trace to
# benchmark/results/allocs/ for hot-path triage. The Piccolissimo alloc-
# profile testitem covers the Altissimo side; this script is the sibling
# for the in-tree NLP solvers.
#
# Uses the same `bilinear_dynamics_and_trajectory` fixture the main test
# suite uses, so the profiled problem is deterministic and small (N=10,
# 4-state × 2-control) — we care about allocation *patterns*, not absolute
# counts on a production-size problem.
#
# Run:
#   julia --project=benchmark benchmark/alloc_profile.jl
# =============================================================================

using Random
using NamedTrajectories
using SparseArrays
using LinearAlgebra
using DirectTrajOpt
using MathOptInterface
const MOI = MathOptInterface
using Ipopt
using MadNLP
using HarmoniqsBenchmarks

# Resolve the MadNLPSolverExt extension module so MadNLPOptions is accessible
# (matches the pattern used in Piccolissimo.jl/benchmark/benchmarks.jl).
const MadNLPSolverExt = [
    mod for mod in reverse(Base.loaded_modules_order)
    if Symbol(mod) == :MadNLPSolverExt
][1]

# Pull in the bilinear fixture without duplicating it.
include(joinpath(@__DIR__, "..", "test", "test_utils.jl"))

Random.seed!(42)

const RESULTS_DIR = joinpath(@__DIR__, "results", "allocs")
mkpath(RESULTS_DIR)

# ----------------------------------------------------------------------------
# Problem builder — wraps the shared fixture with a QuadraticRegularizer-style
# objective so both Ipopt and MadNLP see the same NLP.
# ----------------------------------------------------------------------------
function build_problem(; N = 10)
    G, traj = bilinear_dynamics_and_trajectory(; N = N)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)

    prob = DirectTrajOptProblem(traj, J, integrators)
    return prob, traj
end

# ----------------------------------------------------------------------------
# Profile one solver. Warmup runs on a throwaway deepcopy so JIT/compile
# allocations stay out of the recorded trace.
# ----------------------------------------------------------------------------
function profile_solver(; solver_name, options_ctor, N = 10, sample_rate = 1.0)
    prob_warmup,  traj = build_problem(; N = N)
    prob_profiled, _   = build_problem(; N = N)

    state_dim = traj.dims[:x]
    ctrl_dim  = sum(traj.dims[cn] for cn in traj.control_names if cn != traj.timestep; init = 0)

    println("\n[$(solver_name)] JIT warmup on throwaway problem copy...")
    DirectTrajOpt.solve!(prob_warmup; options = options_ctor())

    println("[$(solver_name)] Profiling allocations (sample_rate=$(sample_rate))...")
    profile = benchmark_memory!(
        package        = "DirectTrajOpt",
        solver         = solver_name,
        benchmark_name = "bilinear_N$(N)_$(lowercase(solver_name))",
        N              = traj.N,
        state_dim      = state_dim,
        control_dim    = ctrl_dim,
        sample_rate    = sample_rate,
        warmup         = false,
        runner         = "local",
    ) do
        DirectTrajOpt.solve!(prob_profiled; options = options_ctor())
    end

    mb = profile.total_bytes / (1024 * 1024)
    println("[$(solver_name)] captured $(profile.total_count) samples, $(round(mb; digits=2)) MB total")

    path = save_alloc_profile(RESULTS_DIR, profile.benchmark_name, profile)
    println("[$(solver_name)] saved to $(path)")
    return profile, path
end

# ----------------------------------------------------------------------------
# Entry points
#
# sample_rate default is 0.01 because Ipopt/MadNLP generate orders of magnitude
# more fine-grained allocations than the solve's wall-time budget accommodates
# at sample_rate=1.0 (an N=10 bilinear toy can hang for 15+ minutes at 1.0).
# 0.01 still gives statistically useful traces for hot-path triage.
# ----------------------------------------------------------------------------
function main(; N = 10, sample_rate = 0.01)
    ipopt_profile, ipopt_path = profile_solver(;
        solver_name   = "Ipopt",
        options_ctor  = () -> IpoptOptions(max_iter = 50, print_level = 0),
        N             = N,
        sample_rate   = sample_rate,
    )

    madnlp_profile, madnlp_path = profile_solver(;
        solver_name   = "MadNLP",
        options_ctor  = () -> MadNLPSolverExt.MadNLPOptions(max_iter = 50, print_level = Int(MadNLP.ERROR)),
        N             = N,
        sample_rate   = sample_rate,
    )

    println("\nDone.")
    println("  Ipopt  profile: $(ipopt_path)  ($(ipopt_profile.total_count) samples)")
    println("  MadNLP profile: $(madnlp_path)  ($(madnlp_profile.total_count) samples)")
    return (ipopt = ipopt_profile, madnlp = madnlp_profile)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
