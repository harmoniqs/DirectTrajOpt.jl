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
    time_name::Symbol=:t,
    timestep_name::Symbol=:Δt,
    label="time consistency constraint (t_{k+1} = t_k + Δt_k)"
)
    return TimeConsistencyConstraint(time_name, timestep_name, label)
end

# =========================================================================== #

@testitem "TimeConsistencyConstraint" begin
    include("../../../test/test_utils.jl")
    using NamedTrajectories

    # Create trajectory with both t and Δt
    N = 10
    Δt_val = 0.1
    times = range(0, step=Δt_val, length=N)
    
    traj = NamedTrajectory(
        (
            x = rand(2, N),
            u = rand(1, N),
            Δt = fill(Δt_val, N),
            t = collect(times)
        );
        controls=(:u, :Δt),
        timestep=:Δt,
        bounds=(u = (-1.0, 1.0),)
    )

    # Create simple problem with time consistency constraint (no dynamics)
    J = QuadraticRegularizer(:u, traj, 1.0)
    
    time_con = TimeConsistencyConstraint()
    
    prob = DirectTrajOptProblem(traj, J, AbstractIntegrator[]; constraints=[time_con])
    solve!(prob; max_iter=100)
    
    # Verify time consistency: t_{k+1} = t_k + Δt_k
    t = prob.trajectory.t
    Δt = prob.trajectory.Δt
    for k in 1:N-1
        @test abs(t[k+1] - t[k] - Δt[k]) < 1e-8
    end
end

@testitem "TimeConsistencyConstraint with free time optimization" begin
    include("../../../test/test_utils.jl")
    using NamedTrajectories

    # Create trajectory with inconsistent t and Δt initially
    N = 10
    Δt_val = 0.5 
    
    traj = NamedTrajectory(
        (
            x = rand(2, N),
            u = rand(1, N),
            Δt = fill(Δt_val, N),
            t = cumsum(rand(N))  # Random times - inconsistent!
        );
        controls=(:u, :Δt, :t),  # t is also a control to be optimized
        timestep=:Δt,
        bounds=(u = (-1.0, 1.0), t = (0.0, 10.0)),
        initial=(t = [0.0],)  # Fix initial time to 0
    )

    # Simple objective - no dynamics needed for this test
    J = QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:t, traj, 0.1)  # Regularize time to encourage consistency
    
    time_con = TimeConsistencyConstraint()
    
    prob = DirectTrajOptProblem(traj, J, AbstractIntegrator[]; constraints=[time_con])
    solve!(prob; max_iter=100)
    
    # Verify time consistency is enforced
    t = prob.trajectory.t
    Δt = prob.trajectory.Δt
    for k in 1:N-1
        @test abs(t[k+1] - t[k] - Δt[k]) < 1e-6
    end
    
    # Verify initial time constraint
    @test abs(t[1]) < 1e-8
end
