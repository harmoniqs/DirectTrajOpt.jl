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

N = 50
traj = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
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
integrator = BilinearIntegrator(G, :x, :u, traj)

# ### Multiple Drives Example

traj_multi = NamedTrajectory(
    (x = randn(3, N), u = randn(2, N), Δt = fill(0.1, N));
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

integrator_multi = BilinearIntegrator(G_multi, :x, :u, traj_multi)

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
        x = randn(2, N),
        u = randn(1, N),
        t = collect(range(0, 5, N)),  # time variable
        Δt = fill(0.1, N)
    );
    timestep=:Δt,
    controls=:u
)

# Time-dependent generator
G_td = (u, t) -> [-0.1 + 0.5*sin(t)  1.0; -1.0  -0.1] + u[1] * [0.0 1.0; 1.0 0.0]

integrator_td = TimeDependentBilinearIntegrator(G_td, :x, :u, :t, traj_td)

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
        x = randn(2, N),
        u = randn(2, N),
        du = zeros(2, N),   # control derivative
        Δt = fill(0.1, N)
    );
    timestep=:Δt,
    controls=:u,
    initial=(u = [0.0, 0.0],),
    final=(u = [0.0, 0.0],)
)

# Enforce du/dt = du
deriv_integrator = DerivativeIntegrator(:u, :du, traj_smooth)

# Now you can penalize `du` to get smooth controls:
# obj = QuadraticRegularizer(:u, traj_smooth, 1e-2)
# obj += QuadraticRegularizer(:du, traj_smooth, 1e-1)  # Smoothness penalty

# ### Multiple Derivative Orders

traj_smooth2 = NamedTrajectory(
    (
        x = randn(2, N),
        u = randn(1, N),
        du = zeros(1, N),
        ddu = zeros(1, N),
        Δt = fill(0.1, N)
    );
    timestep=:Δt,
    controls=:u
)

# Chain derivatives: d(u)/dt = du, d(du)/dt = ddu
deriv_u = DerivativeIntegrator(:u, :du, traj_smooth2)
deriv_du = DerivativeIntegrator(:du, :ddu, traj_smooth2)

# ### When to Use DerivativeIntegrator
# ✓ Enforce smooth, implementable controls
# ✓ Acceleration limits (when control is jerk)
# ✓ Tracking derivative information

# ## Combining Multiple Integrators

# You can use multiple integrators simultaneously:

traj_combined = NamedTrajectory(
    (
        x = randn(2, N),
        u = randn(2, N),
        du = zeros(2, N),
        Δt = fill(0.1, N)
    );
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0], u = [0.0, 0.0]),
    final=(u = [0.0, 0.0],)
)

# Create problem with multiple integrators
G_drift_simple = [-0.1 1.0; -1.0 -0.1]
G_drives_simple = [[0.0 1.0; 1.0 0.0], [1.0 0.0; 0.0 1.0]]
G_simple = u -> G_drift_simple + sum(u .* G_drives_simple)

obj = QuadraticRegularizer(:u, traj_combined, 1e-2)
obj += QuadraticRegularizer(:du, traj_combined, 1e-1)

integrators_combined = [
    BilinearIntegrator(G_simple, :x, :u, traj_combined),
    DerivativeIntegrator(:u, :du, traj_combined)
]

prob = DirectTrajOptProblem(traj_combined, obj, integrators_combined)

# ## Integration Methods Comparison

# | Integrator | Dynamics Type | Accuracy | Use Case |
# |------------|--------------|----------|----------|
# | `BilinearIntegrator` | Control-linear | Exact | Quantum, rotation |
# | `TimeDependentBilinearIntegrator` | Time-varying control-linear | Exact | Modulated systems |
# | `DerivativeIntegrator` | Derivative relation | Exact | Smoothness |

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
# integrator = BilinearIntegrator(G_basic, :x, :u, traj)

# ### Pattern 2: Smooth Control Problem
# integrators = [
#     BilinearIntegrator(G, :x, :u, traj),
#     DerivativeIntegrator(:u, :du, traj)
# ]

# ### Pattern 3: Time-Dependent with Smoothness
# integrators = [
#     TimeDependentBilinearIntegrator(G_td, :x, :u, :t, traj),
#     DerivativeIntegrator(:u, :du, traj)
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
