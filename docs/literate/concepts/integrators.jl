# # Integrators

# ## What are Integrators?

# **Integrators** discretize continuous-time dynamics into constraints for the NLP solver.
# They implement the relationship:
# ```math
# x_{k+1} = \Phi(x_k, u_k, \Delta t_k)
# ```

# where `Φ` approximates the continuous evolution `ẋ = f(x, u, t)`.

using DirectTrajOpt
using NamedTrajectories
using LinearAlgebra

# ## BilinearIntegrator

# ### Overview
# Used for **control-linear** (bilinear) dynamics:
# ```math
# \dot{x} = (G_0 + \sum_i u_i G_i) x
# ```

# where:
# - `G₀` is the drift term (dynamics with no control)
# - `Gᵢ` are the drive terms (how controls affect the system)
# - `uᵢ` are the control inputs

# ### How it Works
# Uses the **matrix exponential** for exact integration:
# ```math
# x_{k+1} = \exp(\Delta t \cdot G(u_k)) x_k
# ```

# where `G(u) = G₀ + Σᵢ uᵢ Gᵢ`.

# ### Example: Simple 2D System

T = 50
traj = NamedTrajectory(
    (x = randn(2, T), u = randn(1, T), Δt = fill(0.1, T));
    timestep=:Δt,
    controls=:u,
    initial=(x = [1.0, 0.0],),
    final=(x = [0.0, 1.0],)
)

# Define drift (natural dynamics) and drives (control terms)
G_drift = [-0.1 1.0; -1.0 -0.1]     # Damped oscillator
G_drives = [[0.0 1.0; 1.0 0.0]]     # Symmetric control coupling

# Create generator function
G = u -> G_drift + sum(u .* G_drives)

# Create integrator
integrator = BilinearIntegrator(G, traj, :x, :u)

# ### Multiple Drives Example

traj_multi = NamedTrajectory(
    (x = randn(3, T), u = randn(2, T), Δt = fill(0.1, T));
    timestep=:Δt,
    controls=:u
)

G_drift_3d = [
    0.0  1.0  0.0;
   -1.0  0.0  0.0;
    0.0  0.0 -0.1
]

G_drives_3d = [
    [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],  # Drive 1
    [0.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0]   # Drive 2
]

G_multi = u -> G_drift_3d + sum(u .* G_drives_3d)

integrator_multi = BilinearIntegrator(G_multi, traj_multi, :x, :u)

# ### When to Use BilinearIntegrator
# ✓ Quantum systems (Hamiltonian evolution)
# ✓ Rotating systems (attitude dynamics)
# ✓ Systems linear in controls
# ✓ When you want exact integration (no discretization error)

# ## TimeDependentBilinearIntegrator

# ### Overview
# For **time-varying** bilinear dynamics:
# ```math
# \dot{x} = (G_0(t) + \sum_i u_i(t) G_i(t)) x
# ```

# The generator function now depends on both control and time.

# ### Example: Periodic Disturbance

traj_td = NamedTrajectory(
    (
        x = randn(2, T),
        u = randn(1, T),
        t = collect(range(0, 5, T)),  # time variable
        Δt = fill(0.1, T)
    );
    timestep=:Δt,
    controls=:u
)

# Time-dependent generator
G_td = (u, t) -> [-0.1 + 0.5*sin(t)  1.0; -1.0  -0.1] + u[1] * [0.0 1.0; 1.0 0.0]

integrator_td = TimeDependentBilinearIntegrator(G_td, traj_td, :x, :u, :t)

# ### When to Use TimeDependentBilinearIntegrator
# ✓ Time-varying Hamiltonians
# ✓ Systems with periodic forcing
# ✓ Carrier wave modulation (e.g., rotating frame transformations)

# ## DerivativeIntegrator

# ### Overview
# Enforces **derivative relationships** between trajectory components:
# ```math
# \frac{d(\text{var})}{dt} = \text{deriv}
# ```

# This is used for smoothness or when controls are derivatives of other variables.

# ### Example: Smooth Controls

traj_smooth = NamedTrajectory(
    (
        x = randn(2, T),
        u = randn(2, T),
        du = zeros(2, T),   # control derivative
        Δt = fill(0.1, T)
    );
    timestep=:Δt,
    controls=:u,
    initial=(u = [0.0, 0.0],),
    final=(u = [0.0, 0.0],)
)

# Enforce du/dt = du
deriv_integrator = DerivativeIntegrator(traj_smooth, :u, :du)

# Now you can penalize `du` to get smooth controls:
# obj = QuadraticRegularizer(:u, traj_smooth, 1e-2)
# obj += QuadraticRegularizer(:du, traj_smooth, 1e-1)  # Smoothness penalty

# ### Multiple Derivative Orders

traj_smooth2 = NamedTrajectory(
    (
        x = randn(2, T),
        u = randn(1, T),
        du = zeros(1, T),
        ddu = zeros(1, T),
        Δt = fill(0.1, T)
    );
    timestep=:Δt,
    controls=:u
)

# Chain derivatives: d(u)/dt = du, d(du)/dt = ddu
deriv_u = DerivativeIntegrator(traj_smooth2, :u, :du)
deriv_du = DerivativeIntegrator(traj_smooth2, :du, :ddu)

# ### When to Use DerivativeIntegrator
# ✓ Enforce smooth, implementable controls
# ✓ Acceleration limits (when control is jerk)
# ✓ Tracking derivative information

# ## TimeIntegrator

# ### Overview
# Manages **time evolution** for the time variable itself:
# ```math
# t_{k+1} = t_k + \Delta t_k
# ```

# Usually only needed when you explicitly track time as a state.

traj_time = NamedTrajectory(
    (
        x = randn(2, T),
        u = randn(1, T),
        t = zeros(1, T),
        Δt = fill(0.1, T)
    );
    timestep=:Δt,
    controls=:u
)

time_integrator = TimeIntegrator(traj_time, :t)

# ### When to Use TimeIntegrator
# ✓ Time-dependent dynamics need explicit time
# ✓ Time-dependent cost functions
# ✓ Tracking total elapsed time

# ## Combining Multiple Integrators

# You can use multiple integrators simultaneously:

traj_combined = NamedTrajectory(
    (
        x = randn(2, T),
        u = randn(2, T),
        du = zeros(2, T),
        t = collect(range(0, 5, T)),
        Δt = fill(0.1, T)
    );
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0], u = [0.0, 0.0]),
    final=(u = [0.0, 0.0],)
)

# Time-varying dynamics
G_combined = (u, t) -> [-0.1 1.0; -1.0 -0.1] + sum(u .* [[0.0 1.0; 1.0 0.0], [1.0 0.0; 0.0 1.0]])

integrators = [
    TimeDependentBilinearIntegrator(G_combined, traj_combined, :x, :u, :t),
    DerivativeIntegrator(traj_combined, :u, :du),
    TimeIntegrator(traj_combined, :t)
]

# Create problem with multiple integrators
G_drift_simple = [-0.1 1.0; -1.0 -0.1]
G_drives_simple = [[0.0 1.0; 1.0 0.0]]
G_simple = u -> G_drift_simple + sum(u .* G_drives_simple)

obj = QuadraticRegularizer(:u, traj_combined, 1e-2)
obj += QuadraticRegularizer(:du, traj_combined, 1e-1)

# Note: Using simpler BilinearIntegrator for this example
integrators_simple = [
    BilinearIntegrator(G_simple, traj_combined, :x, :u),
    DerivativeIntegrator(traj_combined, :u, :du)
]

prob = DirectTrajOptProblem(traj_combined, obj, integrators_simple)

# ## Integration Methods Comparison

# | Integrator | Dynamics Type | Accuracy | Use Case |
# |------------|--------------|----------|----------|
# | `BilinearIntegrator` | Control-linear | Exact | Quantum, rotation |
# | `TimeDependentBilinearIntegrator` | Time-varying control-linear | Exact | Modulated systems |
# | `DerivativeIntegrator` | Derivative relation | Exact | Smoothness |
# | `TimeIntegrator` | Time evolution | Exact | Time tracking |

# ## Custom Integrators

# You can implement custom integrators by subtyping `AbstractIntegrator` and
# defining the constraint function. See the Advanced Topics section for details.

# ### Interface Requirements
# ```julia
# struct MyIntegrator <: AbstractIntegrator
#     # ... fields ...
# end
#
# # Implement constraint evaluation
# function (int::MyIntegrator)(δ, zₖ, zₖ₊₁, k)
#     # Compute constraint: δ = xₖ₊₁ - Φ(xₖ, uₖ, Δtₖ)
#     # where Φ is your integration scheme
# end
# ```

# ## Best Practices

# ### Initialization
# - Start with good initial guesses for states and controls
# - For smooth control problems, initialize derivatives to zero
# - Use linear interpolation for states between boundary conditions

# ### Performance
# - Matrix exponential (BilinearIntegrator) is efficient for small systems (n < 20)
# - For large systems, consider sparse representations
# - DerivativeIntegrator is cheap (just finite differences)

# ### Numerical Stability
# - Keep time steps reasonable (not too large)
# - For stiff systems, smaller time steps help
# - BilinearIntegrator handles stiff systems well

# ## Common Patterns

# ### Pattern 1: Basic Bilinear Problem
G_basic = u -> [-0.1 1.0; -1.0 -0.1] + u[1] * [0.0 1.0; 1.0 0.0]
# integrator = BilinearIntegrator(G_basic, traj, :x, :u)

# ### Pattern 2: Smooth Control Problem  
# integrators = [
#     BilinearIntegrator(G, traj, :x, :u),
#     DerivativeIntegrator(traj, :u, :du)
# ]

# ### Pattern 3: Time-Dependent with Smoothness
# integrators = [
#     TimeDependentBilinearIntegrator(G_td, traj, :x, :u, :t),
#     DerivativeIntegrator(traj, :u, :du),
#     TimeIntegrator(traj, :t)
# ]

# ## Summary

# **Key Takeaways:**
# 1. Integrators convert continuous dynamics to discrete constraints
# 2. BilinearIntegrator is the workhorse for control-linear systems
# 3. DerivativeIntegrator adds smoothness
# 4. You can combine multiple integrators
# 5. Good initialization helps convergence

# ## Next Steps

# - **Objectives**: Learn how to define cost functions
# - **Constraints**: Add bounds and path constraints
# - **Tutorials**: See integrators in complete examples
