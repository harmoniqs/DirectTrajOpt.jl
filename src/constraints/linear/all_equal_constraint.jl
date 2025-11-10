export AllEqualConstraint
export TimeStepsAllEqualConstraint

"""
    struct AllEqualConstraint <: AbstractLinearConstraint

Constraint that all components of a variable should be equal to each other.
Commonly used for fixed timesteps.

# Fields
- `var_name::Symbol`: Variable name to constrain
- `component_index::Int`: Which component of the variable (1 for scalar variables)
- `label::String`: Constraint label
"""
struct AllEqualConstraint <: AbstractLinearConstraint
    var_name::Symbol
    component_index::Int
    label::String
end

"""
    TimeStepsAllEqualConstraint(;label="timesteps all equal constraint")

Constraint that all timesteps are equal (for fixed-timestep trajectories).
The trajectory's timestep variable is inferred when applied.
"""
function TimeStepsAllEqualConstraint(;
    label="timesteps all equal constraint"
)
    # Use a placeholder; actual timestep variable determined from trajectory
    return AllEqualConstraint(:Δt, 1, label)
end

# =========================================================================== #

@testitem "TimeStepsAllEqualConstraint" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u),
        DerivativeIntegrator(:u, :du),
        DerivativeIntegrator(:du, :ddu)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Test fixed timestep constraint
    timesteps_equal_con = TimeStepsAllEqualConstraint()
    
    prob = DirectTrajOptProblem(traj, J, integrators; constraints=[timesteps_equal_con])
    solve!(prob; max_iter=100)

    # Verify all timesteps are equal
    timestep_var = prob.trajectory.timestep
    @assert timestep_var isa Symbol
    
    Δts = [prob.trajectory[t].data[prob.trajectory.components[timestep_var]][1] for t in 1:prob.trajectory.N]
    
    # All timesteps should be equal to the last one
    @test all(abs.(Δts .- Δts[end]) .< 1e-6)
end

@testitem "AllEqualConstraint - custom variable" begin
    include("../../../test/test_utils.jl")

    # Create trajectory with a custom variable to constrain
    N = 10
    traj = NamedTrajectory(
        (
            x = rand(2, N),
            u = rand(1, N),
            a = rand(1, N),  # Custom variable
            Δt = fill(0.1, N)
        );
        controls=(:u, :a),
        timestep=:Δt,
        bounds=(Δt = (0.01, 0.5),),
        initial=(x = [0.0, 0.0],),
        goal=(x = [1.0, 0.0],)
    )

    # Create dynamics function G(u) for bilinear integrator
    G_drift = rand(2, 2)
    G_drive = rand(2, 2)
    G(u) = G_drift + u[1] * G_drive
    integrators = [BilinearIntegrator(G, :x, :u)]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += QuadraticRegularizer(:a, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Constrain all values of variable 'a' to be equal
    a_equal_con = AllEqualConstraint(:a, 1, "all a values equal")
    
    prob = DirectTrajOptProblem(traj, J, integrators; constraints=[a_equal_con])
    solve!(prob; max_iter=100)

    # Verify all 'a' values are equal
    a_vals = [prob.trajectory[t][:a][1] for t in 1:prob.trajectory.N]
    @test all(abs.(a_vals .- a_vals[end]) .< 1e-6)
end
