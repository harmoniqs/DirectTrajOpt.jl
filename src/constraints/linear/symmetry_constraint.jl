export SymmetryConstraint
export SymmetricControlConstraint

"""
    struct SymmetryConstraint <: AbstractLinearConstraint

Constraint enforcing symmetry in trajectory variables across time.
Even symmetry: x[t] = x[N-t+1]
Odd symmetry: x[t] = -x[N-t+1]

# Fields
- `var_name::Symbol`: Variable name to constrain
- `component_indices::Vector{Int}`: Which components of the variable
- `even::Bool`: True for even symmetry (x[t] = x[N-t+1]), false for odd (-x[t] = x[N-t+1])
- `include_timestep::Bool`: Whether to also enforce even symmetry on timesteps
- `label::String`: Constraint label
"""
struct SymmetryConstraint <: AbstractLinearConstraint
    var_name::Symbol
    component_indices::Vector{Int}
    even::Bool
    include_timestep::Bool
    label::String
end

"""
    SymmetricControlConstraint(
        name::Symbol,
        idx::Vector{Int};
        even=true,
        include_timestep=true,
        label="symmetry constraint on \$name"
    )

Constraint enforcing symmetry on control variables.
Indices are computed when applied to a trajectory.
"""
function SymmetricControlConstraint(
    name::Symbol,
    idx::Vector{Int};
    even::Bool = true,
    include_timestep::Bool = true,
    label = "symmetry constraint on $name",
)
    return SymmetryConstraint(name, idx, even, include_timestep, label)
end

# =========================================================================== #

@testitem "SymmetricControlConstraint - even symmetry" begin
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
    J += MinimumTimeObjective(traj)

    # Test even symmetry constraint on control
    sym_constraint =
        SymmetricControlConstraint(:u, [1]; even = true, include_timestep = true)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [sym_constraint])
    solve!(prob; max_iter = 100)

    # Verify even symmetry: u[t] = u[N-t+1]
    N = prob.trajectory.N
    for t = 1:(N÷2)
        u_t = prob.trajectory[t][:u][1]
        u_mirror = prob.trajectory[N-t+1][:u][1]
        @test abs(u_t - u_mirror) < 1e-6
    end

    # Verify timestep symmetry
    timestep_var = prob.trajectory.timestep
    if timestep_var isa Symbol
        for t = 1:(N÷2)
            Δt_t = prob.trajectory[t].data[prob.trajectory.components[timestep_var]][1]
            Δt_mirror =
                prob.trajectory[N-t+1].data[prob.trajectory.components[timestep_var]][1]
            @test abs(Δt_t - Δt_mirror) < 1e-6
        end
    end
end

@testitem "SymmetricControlConstraint - odd symmetry" begin
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
    J += MinimumTimeObjective(traj)

    # Test odd symmetry constraint on control (u[t] = -u[N-t+1])
    sym_constraint =
        SymmetricControlConstraint(:u, [1]; even = false, include_timestep = false)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [sym_constraint])
    solve!(prob; max_iter = 200)

    # Verify odd symmetry: u[t] = -u[N-t+1]
    N = prob.trajectory.N
    for t = 1:(N÷2)
        u_t = prob.trajectory[t][:u][1]
        u_mirror = prob.trajectory[N-t+1][:u][1]
        @test abs(u_t + u_mirror) < 1e-6  # u[t] + u[N-t+1] = 0
    end
end

@testitem "SymmetryConstraint - multiple components" begin
    include("../../../test/test_utils.jl")

    # Create trajectory with multi-dimensional control
    N = 10
    traj = NamedTrajectory(
        (
            x = rand(2, N),
            u = rand(2, N),  # 2D control
            Δt = fill(0.1, N),
        );
        controls = :u,
        timestep = :Δt,
        bounds = (Δt = (0.01, 0.5),),
        initial = (x = [0.0, 0.0],),
        goal = (x = [1.0, 0.0],),
    )

    # Create dynamics function G(u) for bilinear integrator
    G_drift = rand(2, 2)
    G_drives = [rand(2, 2), rand(2, 2)]
    G(u) = G_drift + sum(u .* G_drives)
    integrators = [BilinearIntegrator(G, :x, :u, traj)]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Test symmetry on both components of control
    sym_constraint =
        SymmetricControlConstraint(:u, [1, 2]; even = true, include_timestep = false)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [sym_constraint])
    solve!(prob; max_iter = 100)

    # Verify even symmetry on both components
    N = prob.trajectory.N
    for t = 1:(N÷2)
        u_t = prob.trajectory[t][:u]
        u_mirror = prob.trajectory[N-t+1][:u]
        @test all(abs.(u_t .- u_mirror) .< 1e-6)
    end
end
