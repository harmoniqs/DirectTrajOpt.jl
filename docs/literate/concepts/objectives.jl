# # Objectives

# ## What are Objectives?

# **Objectives** (or cost functions) define what you want to minimize in your optimization problem.
# DirectTrajOpt.jl uses an additive structure where you can combine multiple objective terms:
# ```math
# J_{\text{total}} = w_1 J_1 + w_2 J_2 + \cdots + w_N J_N
# ```

using DirectTrajOpt
using NamedTrajectories
using LinearAlgebra

# Setup a sample trajectory for examples
N = 50
traj = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0],),
    goal=(x = [1.0, 0.0],)
)

# ## QuadraticRegularizer

# ### Overview
# Penalizes the **squared norm** of a variable:
# ```math
# J = \sum_{k=1}^{N} \|v_k\|^2
# ```

# This is the most common objective for regularization.

# ### Control Effort Regularization

obj_u = QuadraticRegularizer(:u, traj, 1.0)
# Minimizes: Σₖ ||uₖ||²

# ### State Regularization

obj_x = QuadraticRegularizer(:x, traj, 0.1)
# Minimizes: 0.1 * Σₖ ||xₖ||²

# ### Control Derivative Regularization (Smoothness)

traj_smooth = NamedTrajectory(
    (x = randn(2, N), u = randn(2, N), du = zeros(2, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u
)

obj_du = QuadraticRegularizer(:du, traj_smooth, 1.0)
# Minimizes: Σₖ ||duₖ||² (encourages smooth controls)

# ### Combining Regularizers

# Typical combination: control effort + smoothness
obj_combined = QuadraticRegularizer(:u, traj_smooth, 1e-2) + 
               QuadraticRegularizer(:du, traj_smooth, 1e-1)
# Small control penalty, larger smoothness penalty

# ### Per-Component Weights

# You can weight each component differently:
obj_weighted = QuadraticRegularizer(:u, traj_smooth, [1.0, 0.5])
# First control component weighted more heavily

# ## MinimumTimeObjective

# ### Overview
# Minimizes the **total trajectory duration**:
# ```math
# J = w \sum_{k=1}^{N} \Delta t_k
# ```

# This encourages fast trajectories.

# ### Basic Usage

obj_time = MinimumTimeObjective(traj, 0.1)
# Minimizes: 0.1 * Σₖ Δtₖ

# ### Time-Energy Tradeoff

# Combine with control regularization to trade off speed vs effort:
obj_tradeoff = QuadraticRegularizer(:u, traj, 1.0) + 
               MinimumTimeObjective(traj, 0.5)
# Higher time weight → faster but more control effort
# Lower time weight → slower but less control effort

# ### Free Time Problems

traj_free_time = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    bounds=(Δt = (0.01, 0.5),)  # Allow variable time steps
)

obj_free_time = QuadraticRegularizer(:u, traj_free_time, 1.0) + 
                MinimumTimeObjective(traj_free_time, 1.0)

# ## TerminalObjective

# ### Overview
# Applies a cost only at the **final time step**:
# ```math
# J = f(x_N)
# ```

# Useful for soft constraints on the final state.

# ### Distance to Goal

x_goal = [1.0, 0.0]
obj_terminal = TerminalObjective(
    x -> norm(x - x_goal)^2,
    :x,
    traj
)
# Penalizes: ||x_N - x_goal||²

# ### Custom Terminal Cost

# Any function of the final state:
obj_custom_terminal = TerminalObjective(
    x -> x[1]^2 + 2*x[2]^2 + x[1]*x[2],
    :x,
    traj
)

# ### When to Use
# - **Soft goal**: Don't enforce exact final state, just penalize deviation
# - **Multiple goals**: Can have terminal costs on multiple variables
# - **Custom metrics**: Use domain-specific final state metrics

# ### Hard vs Soft Constraints

# Hard constraint (via trajectory):
traj_hard = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    final=(x = x_goal,)  # Exact constraint
)

# Soft constraint (via terminal objective):
traj_soft = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    goal=(x = x_goal,)  # For reference only
)
obj_soft = TerminalObjective(x -> 100.0 * norm(x - x_goal)^2, :x, traj_soft)
# Large weight approximates hard constraint

# ## KnotPointObjective

# ### Overview
# Applies a cost at **specific time steps**:
# ```math
# J = \sum_{k \in K} f(x_k, u_k)
# ```

# Useful for waypoints or intermediate constraints.

# ### Single Time Point

obj_knot_single = KnotPointObjective(
    x -> norm(x - [0.5, 0.5])^2,
    :x,
    traj;
    times=[25]  # Only at k=25
)

# ### Multiple Time Points

obj_knot_multi = KnotPointObjective(
    u -> norm(u)^2,
    :u,
    traj;
    times=[10, 20, 30, 40]  # At k=10, 20, 30, 40
)

# ### All Time Points (Path Cost)

obj_knot_all = KnotPointObjective(
    xu -> xu[1]^2 + xu[3]^2,  # xu is concatenated [x; u]
    [:x, :u],
    traj;
    times=1:N  # All time steps
)
# Equivalent to manually summing costs

# ### Waypoint Tracking

waypoints = [
    [0.25, 0.25],  # k=13
    [0.75, 0.75],  # k=38
]
waypoint_times = [13, 38]

obj_waypoints = sum(
    KnotPointObjective(
        x -> 10.0 * norm(x - wp)^2,
        :x,
        traj;
        times=[t]
    )
    for (wp, t) in zip(waypoints, waypoint_times)
)

# ## GlobalObjective

# ### Overview
# Applies a cost to **global variables** (constants across time):
# ```math
# J = f(g)
# ```

# Useful for parameters, scaling factors, or other time-independent variables.

# ### Example with Global Parameter

traj_global = NamedTrajectory(
    (
        x = randn(2, N),
        u = randn(1, N),
        Δt = fill(0.1, N)
    );
    timestep=:Δt,
    controls=:u,
    global_data=[1.0],  # Global parameter
    global_components=(α = 1:1,)
)

obj_global = GlobalObjective(
    α -> (α[1] - 2.0)^2,  # Penalize α deviating from 2
    :α,
    traj_global
)

# ## Combining Objectives

# ### Addition Operator

# The `+` operator combines objectives:
obj1 = QuadraticRegularizer(:u, traj, 1.0)
obj2 = MinimumTimeObjective(traj, 0.1)
obj3 = TerminalObjective(x -> norm(x - x_goal)^2, :x, traj)

obj_total = obj1 + obj2 + obj3

# ### Weighting Strategy

# Common pattern: regularization + task objective + time
obj_pattern = (
    1e-2 * QuadraticRegularizer(:u, traj, 1.0) +      # Small control penalty
    1e-1 * MinimumTimeObjective(traj, 1.0) +          # Moderate time penalty
    1e2 * TerminalObjective(                          # Large goal penalty
        x -> norm(x - x_goal)^2, :x, traj
    )
)

# ### Building Incrementally

obj_build = QuadraticRegularizer(:u, traj, 1.0)

# Add time minimization
obj_build += MinimumTimeObjective(traj, 0.1)

# Add terminal cost
obj_build += TerminalObjective(x -> norm(x - x_goal)^2, :x, traj)

# ## Objective Design Patterns

# ### Pattern 1: Pure Tracking
# Minimize deviation from goal at final time

obj_tracking = TerminalObjective(x -> norm(x - x_goal)^2, :x, traj)

# ### Pattern 2: Energy-Optimal
# Minimize control effort with soft goal

obj_energy = (
    QuadraticRegularizer(:u, traj, 1.0) +
    10.0 * TerminalObjective(x -> norm(x - x_goal)^2, :x, traj)
)

# ### Pattern 3: Minimum-Time
# Fast trajectories with bounded controls

traj_mintime = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    bounds=(u = 1.0, Δt = (0.01, 0.5))
)

obj_mintime = (
    1e-3 * QuadraticRegularizer(:u, traj_mintime, 1.0) +  # Small regularization
    1.0 * MinimumTimeObjective(traj_mintime, 1.0) +       # Minimize time
    100.0 * TerminalObjective(                            # Strong goal
        x -> norm(x - x_goal)^2, :x, traj_mintime
    )
)

# ### Pattern 4: Smooth Control
# Implementable controls with derivative penalties

traj_smooth_obj = NamedTrajectory(
    (x = randn(2, N), u = randn(2, N), du = zeros(2, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    initial=(u = [0.0, 0.0],),
    final=(u = [0.0, 0.0],)
)

obj_smooth_pattern = (
    1e-2 * QuadraticRegularizer(:u, traj_smooth_obj, 1.0) +   # Control effort
    1e-1 * QuadraticRegularizer(:du, traj_smooth_obj, 1.0) +  # Smoothness
    10.0 * TerminalObjective(x -> norm(x - x_goal)^2, :x, traj_smooth_obj)
)

# ### Pattern 5: Waypoint Following
# Hit intermediate points along trajectory

obj_waypoint_pattern = (
    QuadraticRegularizer(:u, traj, 1.0) +
    obj_waypoints +  # From earlier example
    TerminalObjective(x -> norm(x - x_goal)^2, :x, traj)
)

# ## Weight Tuning Guidelines

# ### Relative Magnitudes
# - **Regularization**: 1e-3 to 1e-1 (small, for smoothness/stability)
# - **Task objectives**: 1e0 to 1e2 (primary goal)
# - **Hard constraints via penalties**: 1e2 to 1e4 (large, approximate hard constraints)

# ### Balancing Tradeoffs
# ```julia
# # Fast, aggressive controls
# obj = 1e-4 * QuadraticRegularizer(:u, traj, 1.0) + MinimumTimeObjective(traj, 1.0)
#
# # Slow, gentle controls  
# obj = 1e0 * QuadraticRegularizer(:u, traj, 1.0) + 1e-2 * MinimumTimeObjective(traj, 1.0)
# ```

# ### Iterative Tuning
# 1. Start with task objective only
# 2. Add regularization if needed for stability
# 3. Adjust weights based on results
# 4. Use solver output to guide adjustments

# ## Custom Objectives

# You can create custom objectives by implementing the `Objective` interface.
# All objectives must define how they contribute to the cost and its gradients.

# ### Example Structure (Conceptual)
# ```julia
# # Custom objective for special cost
# my_obj = CustomObjective(params...)
#
# # Add to problem
# obj = QuadraticRegularizer(:u, traj, 1.0) + my_obj
# ```

# See the API Reference for details on implementing custom objectives.

# ## Summary

# | Objective Type | Use Case | Typical Weight |
# |----------------|----------|----------------|
# | `QuadraticRegularizer` | Control effort, smoothness | 1e-3 to 1e-1 |
# | `MinimumTimeObjective` | Fast trajectories | 1e-2 to 1e0 |
# | `TerminalObjective` | Goal reaching | 1e0 to 1e2 |
# | `KnotPointObjective` | Waypoints, path costs | 1e0 to 1e1 |
# | `GlobalObjective` | Parameter penalties | Problem-specific |

# ## Best Practices

# 1. **Start simple**: Use basic regularization + terminal cost first
# 2. **Scale consistently**: Keep objective terms of similar magnitude
# 3. **Use soft constraints**: Prefer high-weight objectives over hard constraints when possible
# 4. **Monitor convergence**: Check if optimizer struggles with certain objectives
# 5. **Iterative refinement**: Adjust weights based on results

# ## Next Steps

# - **Constraints**: Learn about bounds and path constraints
# - **Tutorials**: See complete examples with combined objectives
# - **Problem Setup**: Put it all together to solve optimization problems
