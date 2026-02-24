# # Problem Formulation

# ## Overview

# DirectTrajOpt.jl solves **direct trajectory optimization** problems using 
# **direct transcription**, which converts continuous-time optimal control problems 
# into finite-dimensional nonlinear programs (NLPs).

# ## The General Form

# A trajectory optimization problem has the form:

# ```math
# \begin{align*}
# \underset{x_{1:N}, u_{1:N}}{\text{minimize}} \quad & J(x_{1:N}, u_{1:N}) \\
# \text{subject to} \quad & f(x_{k+1}, x_k, u_k, \Delta t, t_k) = 0, \quad k = 1, \ldots, N-1\\
# & c_k(x_k, u_k) \geq 0, \quad k = 1, \ldots, N \\
# & x_1 = x_{\text{init}}, \quad x_N = x_{\text{goal}} \\
# \end{align*}
# ```

# Let's break down each component:

# ## Decision Variables

# ### States: `x₁, x₂, ..., xₙ`
# The **state** represents the configuration of your system at each time step.
# - For a robot arm: joint angles and velocities
# - For a spacecraft: position and velocity
# - For a quantum system: state vector or unitary operator

# ### Controls: `u₁, u₂, ..., uₙ`
# The **control** (or input) represents what you can actuate.
# - For a robot: motor torques
# - For a spacecraft: thruster forces
# - For quantum systems: electromagnetic field amplitudes

# ### Time Steps: `Δt₁, Δt₂, ..., Δtₙ`
# The **time step** can be:
# - **Fixed**: All Δt are equal and constant
# - **Free**: Each Δt is a decision variable (for minimum time problems)

# ## Cost Function: `J(x, u)`

# The **objective** or **cost function** defines what you want to minimize.
# Common objectives include:

# ### Control Effort
# Minimize energy by penalizing large controls:
# ```math
# J = \sum_{k=1}^{N} \|u_k\|^2
# ```

using DirectTrajOpt
using NamedTrajectories
using LinearAlgebra

# Example:
N = 10
traj = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep = :Δt,
    controls = :u,
)

obj_effort = QuadraticRegularizer(:u, traj, 1.0)

# ### Minimum Time
# Minimize trajectory duration:
# ```math
# J = \sum_{k=1}^{N} \Delta t_k
# ```

obj_time = MinimumTimeObjective(traj, 0.1)  # weight = 0.1

# ### Terminal Cost
# Penalize deviation from goal at final time:
# ```math
# J = \|x_N - x_{\text{goal}}\|^2
# ```

x_goal = [1.0, 0.0]
obj_terminal = TerminalObjective(x -> norm(x - x_goal)^2, :x, traj)

# ### Combined Objectives
# You can add multiple objectives together:

obj_combined = obj_effort + 0.1 * obj_time + 10.0 * obj_terminal

# ## Dynamics Constraints: `f(xₖ₊₁, xₖ, uₖ, Δt, t) = 0`

# The **dynamics constraints** ensure the trajectory obeys the system's equations of motion.
# These are encoded via **integrators** that discretize continuous dynamics.

# ### Continuous Dynamics
# A continuous-time system has the form:
# ```math
# \dot{x}(t) = g(x(t), u(t), t)
# ```

# ### Discrete Approximation
# Direct transcription approximates this using numerical integration:
# ```math
# x_{k+1} \approx \Phi(x_k, u_k, \Delta t)
# ```

# where `Φ` is an integration scheme (e.g., Euler, RK4, matrix exponential).

# ### Example: Bilinear Dynamics
# For control-linear systems:
# ```math
# \dot{x} = (G_0 + \sum_i u_i G_i) x
# ```

# The integrator uses matrix exponential:
# ```math
# x_{k+1} = \exp(\Delta t \cdot G(u_k)) x_k
# ```

G_drift = [-0.1 1.0; -1.0 -0.1]
G_drives = [[0.0 1.0; 1.0 0.0]]
G = u -> G_drift + sum(u .* G_drives)

integrator = BilinearIntegrator(G, :x, :u, traj)

# ## Path Constraints: `c(x, u) ≥ 0`

# **Path constraints** restrict states and controls along the trajectory.

# ### Bounds
# Simple box constraints on variables:
# ```math
# u_{\min} \leq u_k \leq u_{\max}
# ```

traj_bounded = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep = :Δt,
    controls = :u,
    bounds = (u = (-1.0, 1.0),),  # -1 ≤ u ≤ 1
)

# ### Nonlinear Constraints
# More complex constraints (e.g., obstacle avoidance, no-go zones):

# Constraint: keep control magnitude bounded
constraint = NonlinearKnotPointConstraint(
    u -> [1.0 - norm(u)],  # 1 - ||u|| ≥ 0  →  ||u|| ≤ 1
    :u,
    traj;
    equality = false,
)

# ## Boundary Conditions

# ### Initial Condition: `x₁ = x_init`
# Fixes the starting state.

# ### Final Condition: `xₙ = x_goal`
# Fixes the ending state (or penalizes deviation via terminal cost).

traj_bc = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep = :Δt,
    controls = :u,
    initial = (x = [0.0, 0.0],),  # Fixed initial state
    final = (x = [1.0, 0.0],),     # Fixed final state
)

# ## Direct Transcription

# ### Why Direct Transcription?
# - **Mature solvers**: Leverage powerful NLP solvers (Ipopt, SNOPT)
# - **Constraint handling**: Natural way to include path constraints
# - **Warm starting**: Can initialize with good guesses
# - **Large problems**: Scales well to thousands of variables

# ### The NLP Formulation
# After discretization, we have a finite-dimensional problem:
# ```math
# \begin{align*}
# \text{minimize} \quad & J(z) \\
# \text{subject to} \quad & h(z) = 0 \\
# & g(z) \geq 0
# \end{align*}
# ```
# where `z = [x₁, u₁, x₂, u₂, ..., xₙ, uₙ, Δt₁, ..., Δtₙ]` is the decision vector.

# ## When to Use DirectTrajOpt

# DirectTrajOpt.jl is ideal when:
# - ✓ You have smooth dynamics
# - ✓ You need to handle constraints
# - ✓ You want flexibility in cost functions
# - ✓ You can provide reasonable initial guesses

# It may not be ideal when:
# - ✗ Dynamics are highly discontinuous
# - ✗ You need guaranteed global optimality
# - ✗ Real-time performance is critical (use MPC frameworks)

# ## Summary

# | Component | Mathematical Form | Implementation |
# |-----------|------------------|----------------|
# | Decision Variables | `x, u, Δt` | `NamedTrajectory` |
# | Objective | `J(x, u)` | `Objective` (sum of terms) |
# | Dynamics | `f(xₖ₊₁, xₖ, uₖ) = 0` | `AbstractIntegrator` |
# | Path Constraints | `c(x, u) ≥ 0` | `AbstractConstraint` |
# | Boundary Conditions | `x₁ = x_init, xₙ = x_goal` | `initial`, `final` in trajectory |

# ## Next Steps

# - **Trajectories**: Learn how to construct `NamedTrajectory` objects
# - **Integrators**: Understand how dynamics are discretized
# - **Objectives**: Explore different cost functions
# - **Constraints**: Add complex constraints to your problems
