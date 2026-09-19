# Shared helpers for the convergence test items.
# Included by each @testitem via
#     include(joinpath(@__DIR__, "problem_utils.jl"))

"""
    _make_xgate_prob(; N=51, seed=42)

X-gate-style bilinear state-transfer problem: 4D real Bloch-like rep with
`x_init = [1,0,0,0]`, `x_goal = [0,1,0,0]`. Mirrors the shape of
`get_seeded_prob` in `test/solver_test_utils.jl` — terminal cost pulls `x`
toward the goal, and bounds/regularizers are sized so both Ipopt and MadNLP
actually drive infidelity below the convergence target.
"""
function _make_xgate_prob(; N::Int = 51, seed::Int = 42)
    Random.seed!(seed)
    Δt = 0.1
    u_bound = 1.0
    ω = 0.1
    Gx = sparse(Float64[0 0 0 1; 0 0 1 0; 0 -1 0 0; -1 0 0 0])
    Gy = sparse(Float64[0 -1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 1 0])
    Gz = sparse(Float64[0 0 1 0; 0 0 0 -1; -1 0 0 0; 0 1 0 0])
    G(u) = ω * Gz + u[1] * Gx + u[2] * Gy

    x_init = [1.0, 0.0, 0.0, 0.0]
    x_goal = [0.0, 1.0, 0.0, 0.0]

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
        bounds = (u = (-u_bound, u_bound), Δt = (Δt, Δt)),
        initial = (x = x_init, u = zeros(2)),
        final = (u = zeros(2),),
        goal = (x = x_goal,),
    )
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> 1e3 * sum(abs2, x - x_goal), :x, traj)
    J += QuadraticRegularizer(:u, traj, 1e-2)
    J += QuadraticRegularizer(:du, traj, 1e-2)
    return DirectTrajOptProblem(traj, J, integrators)
end

"""
    _xgate_infidelity(prob) -> Float64

Infidelity = 1 - <x_final, x_goal>, clamped to [0, 1]. Cheap because both
vectors are unit-norm in this representation.
"""
_xgate_infidelity(prob) = clamp(
    1.0 - LinearAlgebra.dot(prob.trajectory.x[:, end], [0.0, 1.0, 0.0, 0.0]),
    0.0,
    1.0,
)

"""
    _build_convergence_result(result::BenchmarkResult, crit::ConvergenceCriterion;
                              iterations::Union{Nothing,Int}=nothing)

Return a copy of `result` with `crit` attached as its `convergence` field
(optionally overriding `iterations`). `BenchmarkResult` is immutable, so we
rebuild it positionally; pulling this out of the testitems keeps them
readable.
"""
function _build_convergence_result(
    result::HarmoniqsBenchmarks.BenchmarkResult,
    crit::HarmoniqsBenchmarks.ConvergenceCriterion;
    iterations::Union{Nothing,Int} = nothing,
)
    iters = iterations === nothing ? result.iterations : iterations
    return HarmoniqsBenchmarks.BenchmarkResult(
        package = result.package,
        package_version = result.package_version,
        commit = result.commit,
        benchmark_name = result.benchmark_name,
        N = result.N,
        state_dim = result.state_dim,
        control_dim = result.control_dim,
        n_constraints = result.n_constraints,
        n_variables = result.n_variables,
        wall_time_s = result.wall_time_s,
        iterations = iters,
        objective_value = result.objective_value,
        constraint_violation = result.constraint_violation,
        solver_status = result.solver_status,
        solver = result.solver,
        total_allocations_bytes = result.total_allocations_bytes,
        total_allocs_count = result.total_allocs_count,
        gc_time_ns = result.gc_time_ns,
        gc_count = result.gc_count,
        gc_full_count = result.gc_full_count,
        peak_rss_delta_bytes = result.peak_rss_delta_bytes,
        live_heap_delta_bytes = result.live_heap_delta_bytes,
        oom_margin_bytes = result.oom_margin_bytes,
        solver_options = result.solver_options,
        convergence = crit,
        julia_version = result.julia_version,
        timestamp = result.timestamp,
        runner = result.runner,
        n_threads = result.n_threads,
    )
end
