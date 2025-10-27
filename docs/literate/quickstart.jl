# # Quickstart Guide

# Welcome to **DirectTrajOpt.jl**! This guide will get you up and running in minutes.

# ## What is DirectTrajOpt?

# DirectTrajOpt.jl solves **trajectory optimization problems** - finding optimal control sequences
# that drive a dynamical system from an initial state to a goal state while minimizing a cost function.

# ## Installation

# First, install the package:

# ```julia
# using Pkg
# Pkg.add("DirectTrajOpt")
# ```

# You'll also need NamedTrajectories.jl for defining trajectories:

using DirectTrajOpt
using NamedTrajectories
using LinearAlgebra

# ## A Minimal Example

# Let's solve a simple problem: drive a 2D system from `[0, 0]` to `[1, 0]` 
# with minimal control effort.

# ### Step 1: Define the Trajectory

# A trajectory contains your states, controls, and time information:

N = 50  # number of time steps
traj = NamedTrajectory(
    (
        x = randn(2, N),    # 2D state
        u = randn(1, N),    # 1D control
        Δt = fill(0.1, N)   # time step
    );
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0],),
    final=(x = [1.0, 0.0],)
)

# ### Step 2: Define the Dynamics

# Specify how your system evolves. For bilinear dynamics `ẋ = (G₀ + u₁G₁) x`:

G_drift = [-0.1 1.0; -1.0 -0.1]   # drift term
G_drives = [[0.0 1.0; 1.0 0.0]]   # control term
G = u -> G_drift + sum(u .* G_drives)

integrator = BilinearIntegrator(G, traj, :x, :u)

# ### Step 3: Define the Objective

# What do we want to minimize? Let's penalize control effort:

obj = QuadraticRegularizer(:u, traj, 1.0)

# ### Step 4: Create and Solve

# Combine everything into a problem and solve:

prob = DirectTrajOptProblem(traj, obj, integrator)
solve!(prob; max_iter=100, verbose=false)

# ### Step 5: Access the Solution

# The optimized trajectory is stored in `prob.trajectory`:

println("Final state: ", prob.trajectory.x[:, end])
println("Control norm: ", norm(prob.trajectory.u))

# ## What You Can Do

# - **Multiple objectives**: Combine regularization, minimum time, terminal costs
# - **Flexible dynamics**: Linear, bilinear, time-dependent systems
# - **Add constraints**: Bounds, path constraints, custom nonlinear constraints
# - **Smooth controls**: Penalize derivatives for smooth, implementable controls
# - **Free time**: Optimize trajectory duration

# ## Next Steps

# - **Core Concepts**: Understand trajectories, integrators, objectives, and constraints in depth
# - **Tutorials**: Work through progressively complex examples
# - **API Reference**: Explore all available features

# ---

# That's it! You've solved your first trajectory optimization problem. 
# Check out the tutorials to learn more advanced techniques.
