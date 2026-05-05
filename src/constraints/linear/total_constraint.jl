export TotalConstraint
export DurationConstraint

"""
    struct TotalConstraint <: AbstractLinearConstraint

Constraint that the sum of a variable's components equals a target value.
Commonly used for trajectory duration constraints.

# Fields
- `var_name::Symbol`: Variable name to sum
- `component_index::Int`: Which component of the variable (1 for scalar variables)
- `value::Float64`: Target sum value
- `label::String`: Constraint label

# Note
When applied to the trajectory's timestep variable, only the first N-1 timesteps are summed
(the last knot point has no duration after it). For other variables, all N values are summed.
"""
struct TotalConstraint <: AbstractLinearConstraint
    var_name::Symbol
    component_index::Int
    value::Float64
    label::String
end

"""
    DurationConstraint(value::Float64; label="duration constraint of \$value")

Constraint that the total trajectory duration equals a target value.
The trajectory's timestep variable is inferred when applied.

# Note
Duration is computed as the sum of the first N-1 timesteps, since the final knot point
represents the end state and has no duration after it.
"""
function DurationConstraint(value::Float64; label = "duration constraint of $value")
    # Use placeholder; actual timestep variable determined from trajectory
    return TotalConstraint(:Δt, 1, value, label)
end

function Base.show(io::IO, c::TotalConstraint)
    print(io, "TotalConstraint: \"$(c.label)\"")
end

# =========================================================================== #

@testitem "DurationConstraint" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:du, traj, 1.0)

    # Test duration constraint
    target_duration = 4.0
    dur_constraint = DurationConstraint(target_duration)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [dur_constraint])
    solve!(prob; max_iter = 100)

    # Verify total duration equals target
    duration = get_duration(prob.trajectory)
    @test abs(duration - target_duration) < 1e-6
end

@testitem "TotalConstraint - custom variable" begin
    include("../../../test/test_utils.jl")

    # Create trajectory with a custom variable to sum
    N = 10
    traj = NamedTrajectory(
        (
            x = rand(2, N),
            u = rand(1, N),
            w = rand(1, N),  # Custom variable to sum
            Δt = fill(0.1, N),
        );
        controls = (:u, :w),
        timestep = :Δt,
        bounds = (Δt = (0.01, 0.5),),
        initial = (x = [0.0, 0.0],),
        goal = (x = [1.0, 0.0],),
    )

    # Create dynamics function G(u) for bilinear integrator
    G_drift = rand(2, 2)
    G_drive = rand(2, 2)
    G(u) = G_drift + u[1] * G_drive
    integrators = [BilinearIntegrator(G, :x, :u, traj)]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:w, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Constrain sum of 'w' values
    target_sum = 5.0
    total_con = TotalConstraint(:w, 1, target_sum, "sum of w = $target_sum")

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [total_con])
    solve!(prob; max_iter = 100)

    # Verify sum equals target
    w_vals = [prob.trajectory[t][:w][1] for t = 1:prob.trajectory.N]
    @test abs(sum(w_vals) - target_sum) < 1e-6
end
