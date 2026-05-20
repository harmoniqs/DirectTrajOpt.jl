export TimeConsistencyConstraint

"""
    struct TimeConsistencyConstraint <: AbstractLinearConstraint

Constraint that enforces consistency between time values and timesteps:
    t_{k+1} = t_k + Δt_k  for k = 1, ..., T-1

This is used when both absolute times (`:t`) and timesteps (`:Δt`) are stored
in the trajectory and need to remain consistent during optimization.

# Fields
- `time_name::Symbol`: Name of the time variable (default `:t`)
- `timestep_name::Symbol`: Name of the timestep variable (default `:Δt`)
- `label::String`: Constraint label
"""
struct TimeConsistencyConstraint <: AbstractLinearConstraint
    time_name::Symbol
    timestep_name::Symbol
    label::String
end

"""
    TimeConsistencyConstraint(;
        time_name::Symbol=:t,
        timestep_name::Symbol=:Δt,
        label="time consistency constraint (t_{k+1} = t_k + Δt_k)"
    )

Construct a constraint enforcing t_{k+1} = t_k + Δt_k for all k.

# Arguments
- `time_name`: Name of the time variable in the trajectory (default `:t`)
- `timestep_name`: Name of the timestep variable in the trajectory (default `:Δt`)
- `label`: Constraint label for logging/debugging
"""
function TimeConsistencyConstraint(;
    time_name::Symbol = :t,
    timestep_name::Symbol = :Δt,
    label = "time consistency constraint (t_{k+1} = t_k + Δt_k)",
)
    return TimeConsistencyConstraint(time_name, timestep_name, label)
end

function Base.show(io::IO, c::TimeConsistencyConstraint)
    print(
        io,
        "TimeConsistencyConstraint: $(c.time_name)_{k+1} = $(c.time_name)_k + $(c.timestep_name)_k",
    )
end

# =========================================================================== #

@testitem "TimeConsistencyConstraint" begin
    include("../../../test/test_utils.jl")
    using NamedTrajectories

    # Create trajectory with both t and Δt
    N = 10
    Δt_val = 0.1
    times = range(0, step = Δt_val, length = N)

    traj = NamedTrajectory(
        (x = rand(2, N), u = rand(1, N), Δt = fill(Δt_val, N), t = collect(times));
        controls = (:u, :Δt),
        timestep = :Δt,
        bounds = (u = (-1.0, 1.0),),
    )

    # Create simple problem with time consistency constraint (no dynamics)
    J = QuadraticRegularizer(:u, traj, 1.0)

    time_con = TimeConsistencyConstraint()

    prob = DirectTrajOptProblem(traj, J, AbstractIntegrator[]; constraints = [time_con])
    solve!(prob; max_iter = 100)

    # Verify time consistency: t_{k+1} = t_k + Δt_k
    t = prob.trajectory.t
    Δt = prob.trajectory.Δt
    for k = 1:(N-1)
        @test abs(t[k+1] - t[k] - Δt[k]) < 1e-8
    end
end

@testitem "TimeConsistencyConstraint with free time optimization" begin
    include("../../../test/test_utils.jl")
    using NamedTrajectories
    using Random

    # Deterministic seed: same trajectory + multipliers on every run. A failure
    # here is a real regression, not RNG drift. Robustness across seeds is
    # covered by the `:robustness` testitem below.
    rng = MersenneTwister(0)

    # Create trajectory with inconsistent t and Δt initially
    N = 10
    Δt_val = 0.5

    traj = NamedTrajectory(
        (
            x = rand(rng, 2, N),
            u = rand(rng, 1, N),
            Δt = fill(Δt_val, N),
            t = cumsum(rand(rng, N)),  # Random times - inconsistent!
        );
        controls = (:u, :Δt, :t),  # t is also a control to be optimized
        timestep = :Δt,
        bounds = (u = (-1.0, 1.0), t = (0.0, 10.0)),
        initial = (t = [0.0],),  # Fix initial time to 0
    )

    # Simple objective - no dynamics needed for this test
    J = QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:t, traj, 0.1)  # Regularize time to encourage consistency

    time_con = TimeConsistencyConstraint()

    prob = DirectTrajOptProblem(traj, J, AbstractIntegrator[]; constraints = [time_con])
    solve!(prob; max_iter = 100)

    # Verify time consistency is enforced
    t = prob.trajectory.t
    Δt = prob.trajectory.Δt
    for k = 1:(N-1)
        @test abs(t[k+1] - t[k] - Δt[k]) < 1e-6
    end

    # Verify initial time constraint
    @test abs(t[1]) < 1e-8
end

@testitem "TimeConsistencyConstraint free-time robustness sweep" begin
    include("../../../test/test_utils.jl")
    using NamedTrajectories
    using Random

    # K independent seeds; pass if ≥80% land within the same tolerance the
    # deterministic test uses. Catches regressions where the solver's local-
    # minimum behavior degrades for "typical" inconsistent initializations.
    function run_sweep(K::Int)
        pass_count = 0
        failures = String[]
        for seed = 1:K
            rng = MersenneTwister(seed)
            N = 10
            Δt_val = 0.5

            traj = NamedTrajectory(
                (
                    x = rand(rng, 2, N),
                    u = rand(rng, 1, N),
                    Δt = fill(Δt_val, N),
                    t = cumsum(rand(rng, N)),
                );
                controls = (:u, :Δt, :t),
                timestep = :Δt,
                bounds = (u = (-1.0, 1.0), t = (0.0, 10.0)),
                initial = (t = [0.0],),
            )

            J = QuadraticRegularizer(:u, traj, 1.0)
            J += QuadraticRegularizer(:t, traj, 0.1)

            time_con = TimeConsistencyConstraint()

            prob = DirectTrajOptProblem(
                traj,
                J,
                AbstractIntegrator[];
                constraints = [time_con],
            )

            ok = try
                solve!(prob; max_iter = 100)
                t = prob.trajectory.t
                Δt = prob.trajectory.Δt
                consistent = all(k -> abs(t[k+1] - t[k] - Δt[k]) < 1e-6, 1:(N-1))
                initial_ok = abs(t[1]) < 1e-8
                consistent && initial_ok
            catch e
                push!(failures, "seed $seed: threw $(typeof(e))")
                false
            end

            if ok
                pass_count += 1
            else
                push!(failures, "seed $seed: tolerance not met after solve")
            end
        end
        return pass_count, failures
    end

    K = 20
    pass_threshold = 0.80
    pass_count, failures = run_sweep(K)
    pass_rate = pass_count / K
    @info "TimeConsistencyConstraint free-time robustness sweep" pass_count K pass_rate failures
    @test pass_rate >= pass_threshold
end
