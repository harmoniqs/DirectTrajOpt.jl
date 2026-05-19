# # Constraints

# Constraints restrict the feasible region beyond dynamics. DirectTrajOpt supports bounds,
# boundary conditions, and nonlinear path constraints.

using DirectTrajOpt
using NamedTrajectories
using LinearAlgebra

N = 50
traj = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep = :Δt,
    controls = :u,
)

# ## Bounds (Cheapest)

# Box constraints on variables:

traj_bounds = NamedTrajectory(
    (x = randn(2, N), u = randn(2, N), Δt = fill(0.1, N));
    timestep = :Δt,
    controls = :u,
    bounds = (
        x = 5.0,                          # -5 ≤ x ≤ 5
        u = (-1.0, 2.0),                  # -1 ≤ u ≤ 2
        Δt = (0.01, 0.5),                  # 0.01 ≤ Δt ≤ 0.5
    ),
)

# Per-component bounds:
traj_component_bounds = NamedTrajectory(
    (x = randn(2, N), u = randn(2, N), Δt = fill(0.1, N));
    timestep = :Δt,
    controls = :u,
    bounds = (u = ([-1.0, -2.0], [1.0, 3.0]),),  # Different bounds per component
)

# ## Nonlinear Constraints

# **Inequality**: `c(x, u) ≥ 0` (preferred - easier to satisfy)
constraint_ineq = NonlinearKnotPointConstraint(
    u -> [1.0 - norm(u)],  # ||u|| ≤ 1
    :u,
    traj;
    times = 1:N,
    equality = false,
)

# **Equality**: `c(x, u) = 0` (more restrictive)
constraint_eq = NonlinearKnotPointConstraint(
    x -> [x[1] - 0.5],  # x₁ = 0.5
    :x,
    traj;
    times = [25],
    equality = true,
)

# **Multiple variables**:
constraint_multi = NonlinearKnotPointConstraint(
    (x, u) -> [x[1]^2 + x[2]^2 - u[1]],
    [:x, :u],
    traj;
    equality = false,
)

# ## Common Patterns

# **Obstacle avoidance**:
obs_center, obs_radius = [0.5, 0.5], 0.2
constraint_obstacle = NonlinearKnotPointConstraint(
    x -> [norm(x - obs_center)^2 - obs_radius^2],
    :x,
    traj;
    times = 1:N,
    equality = false,
)

# **Multiple obstacles**:
constraints_obstacles = [
    NonlinearKnotPointConstraint(
        x -> [norm(x - center)^2 - radius^2],
        :x,
        traj;
        equality = false,
    ) for (center, radius) in [([0.3, 0.3], 0.15), ([0.7, 0.7], 0.15)]
]

# **State-dependent control limits**:
constraint_state_dep = NonlinearKnotPointConstraint(
    (x, u) -> [1.0 - u[1] / (1.0 + abs(x[1]))],
    [:x, :u],
    traj;
    equality = false,
)

# **Energy constraints**:
E_max = 2.0
constraint_energy = NonlinearKnotPointConstraint(
    (x, u) -> [E_max - (0.5 * norm(x)^2 + 0.5 * norm(u)^2)],
    [:x, :u],
    traj;
    equality = false,
)

# ## Time Selection

# All times, specific times, or ranges:
constraint_all = NonlinearKnotPointConstraint(
    u -> [1.0 - norm(u)],
    :u,
    traj;
    times = 1:N,
    equality = false,
)

constraint_specific = NonlinearKnotPointConstraint(
    x -> [x[1]^2 + x[2]^2 - 1.0],
    :x,
    traj;
    times = [1, 10, 20, 30, 40, 50],
    equality = false,
)

constraint_range = NonlinearKnotPointConstraint(
    u -> [0.5 - norm(u)],
    :u,
    traj;
    times = 10:40,
    equality = false,
)

# ## Creating a Problem

G_drift = [-0.1 1.0; -1.0 -0.1]
G_drives = [[0.0 1.0; 1.0 0.0]]
G = u -> G_drift + sum(u .* G_drives)
integrator = BilinearIntegrator(G, :x, :u, traj)
obj = QuadraticRegularizer(:u, traj, 1.0)

constraints = [constraint_obstacle, constraint_ineq]
prob = DirectTrajOptProblem(traj, obj, integrator; constraints = constraints)

# ## Summary

# | Constraint Type | Form | Cost | Use Case |
# |----------------|------|------|----------|
# | Bounds | `l ≤ v ≤ u` | Very cheap | Physical limits |
# | Dynamics | `xₖ₊₁ = Φ(xₖ, uₖ)` | Moderate | System evolution |
# | Boundary | `x₁ = x₀, xₖ = xf` | Cheap | Initial/final states |
# | Nonlinear inequality | `c(x, u) ≥ 0` | Moderate | Obstacles, limits |
# | Nonlinear equality | `c(x, u) = 0` | Expensive | Exact requirements |

# ## Performance Tips

# | Recommendation | Rationale |
# |----------------|-----------|
# | Use bounds over nonlinear constraints | Much faster to evaluate |
# | Prefer inequalities over equalities | Easier to satisfy, larger feasible region |
# | Scale constraint values to O(1) | Better numerical conditioning |
# | Add constraints incrementally | Easier to debug, avoids over-constraining |
# | Check initial guess feasibility | Prevents infeasible starts |

# ## Troubleshooting

# If optimizer struggles:
# - **Infeasible start**: Initial guess violates constraints → improve initial guess
# - **Over-constrained**: Too many/conflicting constraints → relax or remove some
# - **Poorly scaled**: Values span many orders of magnitude → rescale to O(1)
# - **Tight constraints**: Little feasible space → relax bounds or use soft constraints
