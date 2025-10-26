# # DirectTrajOpt.jl Guide

# ## Overview

# **DirectTrajOpt.jl** provides a framework for solving direct trajectory optimization problems.
# It uses direct transcription to convert continuous optimal control problems into nonlinear 
# programs (NLPs) that can be solved with gradient-based optimization.

# ## Installation

using DirectTrajOpt
using NamedTrajectories

# ## Problem Formulation

# DirectTrajOpt solves problems of the form:

# ```math
# \begin{align*}
# \underset{x_{1:T}, u_{1:T}}{\text{minimize}} \quad & J(x_{1:T}, u_{1:T}) \\
# \text{subject to} \quad & f(x_{k+1}, x_k, u_k, \Delta t, t_k) = 0, \quad k = 1, \ldots, T-1\\
# & c_k(x_k, u_k) \geq 0, \quad k = 1, \ldots, T \\
# & x_1 = x_{\text{init}}, \quad x_T = x_{\text{goal}} \\
# \end{align*}
# ```

# where:
# - `J` is the objective function to minimize
# - `f` represents the system dynamics (implemented via integrators)
# - `c` represents additional constraints
# - `x` is the state trajectory
# - `u` is the control trajectory

# ## Basic Usage

# ### 1. Define a Trajectory

# Create a NamedTrajectory with your state and control variables:

T = 50  # number of time steps
n_states = 2
n_controls = 1

traj = NamedTrajectory(
    (
        x = randn(n_states, T),
        u = randn(n_controls, T),
        Δt = fill(0.1, T)
    );
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0],),
    final=(x = [1.0, 0.0],)
)

# ### 2. Define Dynamics

# Use integrators to specify how your system evolves:

# Example: Linear dynamics ẋ = (G₀ + Σ uᵢ Gᵢ) x 
G_drift = [-0.1 1.0; -1.0 -0.1]
G_drives = [
    [0.0 1.0; 1.0 0.0]
]
G = u -> G_drift + sum(u .* G_drives)

integrator = BilinearIntegrator(G, traj, :x, :u)

# ### 3. Define Objectives

# Specify what you want to minimize:

# Minimize control effort
obj = QuadraticRegularizer(:u, traj, 1.0)

# Add minimum time objective
obj += MinimumTimeObjective(traj; D=0.1)

# ### 4. Create and Solve the Problem

prob = DirectTrajOptProblem(traj, obj, integrator)

solve!(prob; max_iter=100, verbose=true)

# The solution is now stored in `prob.trajectory`.

# ## Key Components

# ### Integrators

# Integrators define the dynamics constraints. Available integrators include:
# - `BilinearIntegrator`: For control-linear dynamics
# - `TimeDependentBilinearIntegrator`: For time-varying dynamics  
# - `DerivativeIntegrator`: For enforcing smoothness
# - `TimeIntegrator`: For time evolution

# ### Objectives

# Combine multiple objective terms:
# - `QuadraticRegularizer`: Penalize large control/state values
# - `MinimumTimeObjective`: Minimize trajectory duration
# - `TerminalObjective`: Cost on final state
# - `KnotPointObjective`: Cost at specific time points
# - `GlobalObjective`: Cost on global variables

# ### Constraints

# Add constraints beyond dynamics:
# - `BoundsConstraint`: Variable bounds
# - `EqualityConstraint`: Equality constraints  
# - `NonlinearConstraint`: General nonlinear constraints
# - `LinearConstraint`: Linear constraints

# ## Advanced Example

# Create a more complex problem with multiple objectives and constraints:

# Define trajectory with bounds (including derivative variable)
traj_advanced = NamedTrajectory(
    (
        x = randn(2, T),
        u = randn(1, T),
        du = zeros(1, T),
        Δt = fill(0.1, T)
    );
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0],),
    final=(x = [1.0, 0.0],),
    bounds=(u = (-1.0, 1.0),)
)

# Redefine dynamics for the advanced trajectory
G_drift_advanced = [-0.1 1.0; -1.0 -0.1]
G_drives_advanced = [
    [0.0 1.0; 1.0 0.0]
]
G_advanced = u -> G_drift_advanced + sum(u .* G_drives_advanced)

# Multiple integrators for different dynamics
integrators = [
    BilinearIntegrator(G_advanced, traj_advanced, :x, :u),
    DerivativeIntegrator(traj_advanced, :u, :du)  # enforce smooth controls
]

# Combined objective
obj_advanced = QuadraticRegularizer(:u, traj_advanced, 1e-2)
obj_advanced += QuadraticRegularizer(:du, traj_advanced, 1e-1)
obj_advanced += MinimumTimeObjective(traj_advanced; D=0.1)

# Create and solve
prob_advanced = DirectTrajOptProblem(traj_advanced, obj_advanced, integrators)
solve!(prob_advanced; max_iter=200)

# ## Next Steps

# - Explore the Library documentation for detailed API reference
# - Check out the test files for more examples
# - Customize integrators and objectives for your specific problem

