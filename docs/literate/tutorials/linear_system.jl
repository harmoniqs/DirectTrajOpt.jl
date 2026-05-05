# # Tutorial: Linear System Control

# In this tutorial, we'll solve a simple 2D linear control problem from start to finish.

# ## Problem Description

# We want to control a 2D oscillator from rest at the origin to a target position,
# minimizing control effort.

# **System dynamics:**
# ```math
# \dot{x} = (G_0 + u_1 G_1) x
# ```

# **Goal:** Drive from `x(0) = [0, 0]` to `x(N) = [1, 0]`

# **Objective:** Minimize control effort `∫ ||u||² dt`

using DirectTrajOpt
using NamedTrajectories
using LinearAlgebra
using Statistics
using Printf

# ## Step 1: Define the System Dynamics

# The drift matrix (natural dynamics):
G_drift = [
    -0.1 1.0;
    -1.0 -0.1
]

# The drive matrix (control influence):
G_drives = [[
    0.0 1.0;
    1.0 0.0
]]

# Generator function:
G = u -> G_drift + sum(u .* G_drives)

# Let's understand what this system does:
# - G_drift creates damped oscillations
# - G_drives couples the two states symmetrically
# - Control u affects how the states influence each other

# ## Step 2: Create the Trajectory

# Time parameters
N = 50# number of time steps
Δt = 0.1        # time step size
total_time = N * Δt  # 5 seconds

println("Total time: $total_time seconds")

# Initial and goal states
x_init = [0.0, 0.0]
x_goal = [1.0, 0.0]

# Create initial guess with linear interpolation
x_guess = hcat([x_init + (x_goal - x_init) * (t/(N-1)) for t = 0:(N-1)]...)
u_guess = zeros(1, N)

# Create the trajectory
traj = NamedTrajectory(
    (x = x_guess, u = u_guess, Δt = fill(Δt, N));
    timestep = :Δt,
    controls = :u,
    initial = (x = x_init,),
    final = (x = x_goal,),
)

println("Trajectory dimensions:")
println("  States: ", traj.dims.x)
println("  Controls: ", traj.dims.u)
println("  Time steps: ", traj.N)

# ## Step 3: Define the Dynamics Constraint

# Use BilinearIntegrator for our control-linear system:
integrator = BilinearIntegrator(G, :x, :u, traj)

println("Integrator created for bilinear dynamics")

# ## Step 4: Define the Objective

# Minimize control effort:
obj = QuadraticRegularizer(:u, traj, 1.0)

println("Objective: minimize ∫ ||u||² dt")

# ## Step 5: Create and Solve the Problem

# Assemble the optimization problem:
prob = DirectTrajOptProblem(traj, obj, integrator)

prob

#-

println("Solving optimization problem...")
println("="^50)

# Solve with Ipopt:
solve!(prob; max_iter = 100, verbose = false)

println("="^50)
println("Optimization complete!\n")

# ## Step 6: Analyze the Solution

# Extract the solution:
x_sol = prob.trajectory.x
u_sol = prob.trajectory.u
times = cumsum([0.0; prob.trajectory.Δt[:]])

println("Solution analysis:")
println("  Initial state: ", x_sol[:, 1])
println("  Final state: ", x_sol[:, end])
println("  Goal state: ", x_goal)
println("  Final error: ", norm(x_sol[:, end] - x_goal))

# Control statistics:
u_norm = norm(u_sol)
u_max = maximum(abs.(u_sol))
u_mean = mean(abs.(u_sol))

println("\nControl statistics:")
println("  Total norm: ", u_norm)
println("  Max magnitude: ", u_max)
println("  Mean magnitude: ", u_mean)

# ## Step 7: Verify Dynamics

# Check that the solution satisfies the dynamics at a few points:

function verify_dynamics(x, u, Δt, G, k)
    ## Compute x[k+1] using the dynamics
    x_k = x[:, k]
    u_k = u[:, k]
    Δt_k = Δt[k]
    ## Matrix exponential integration
    x_k1_predicted = exp(Δt_k * G(u_k)) * x_k
    x_k1_actual = x[:, k+1]

    error = norm(x_k1_predicted - x_k1_actual)
    return error
end

println("\nDynamics verification (error at selected time steps):")
for k in [1, 10, 25, 40, N-1]
    error = verify_dynamics(x_sol, u_sol, prob.trajectory.Δt, G, k)
    println("  k=$k: error = ", error)
end

# ## Visualization (Conceptual)

# In a Jupyter notebook or with plotting packages, you could visualize:

println("\n" * "="^50)
println("SOLUTION SUMMARY")
println("="^50)

println("\nState trajectory (first 10 and last 10 time steps):")
println("Time | x₁      | x₂")
println("-"^25)
for k in [1:10; (N-9):N]
    t = times[k]
    println(@sprintf("%.2f | %7.4f | %7.4f", t, x_sol[1, k], x_sol[2, k]))
end

println("\nControl trajectory (first 10 and last 10 time steps):")
println("Time | u")
println("-"^15)
for k in [1:10; (N-9):N]
    t = times[k]
    println(@sprintf("%.2f | %7.4f", t, u_sol[1, k]))
end

# ## Key Takeaways

# 1. **Linear interpolation** provides a good initial guess for smooth problems
# 2. **BilinearIntegrator** handles control-linear dynamics exactly
# 3. **Boundary conditions** (initial/final) are enforced as hard constraints
# 4. **Control effort minimization** produces smooth, efficient controls
# 5. The solver finds a solution that **satisfies dynamics** and **reaches the goal**

# ## Exercises

# Try modifying the problem:

# ### Exercise 1: Change the goal
# Try reaching `x_goal = [0.5, 0.5]` instead

# ### Exercise 2: Add control bounds
# Limit the control: `-1.0 ≤ u ≤ 1.0`
# ```julia
# traj_bounded = NamedTrajectory(
#     (x = x_guess, u = u_guess, Δt = fill(Δt, N));
#     timestep=:Δt,
#     controls=:u,
#     initial=(x = x_init,),
#     final=(x = x_goal,),
#     bounds=(u = 1.0,)
# )
# ```

# ### Exercise 3: Vary the control weight
# Try `QuadraticRegularizer(:u, traj, 0.1)` (less penalty) or 
# `QuadraticRegularizer(:u, traj, 10.0)` (more penalty)

# ### Exercise 4: Add a terminal cost
# Use soft goal constraint instead:
# ```julia
# traj_soft = NamedTrajectory(
#     (x = x_guess, u = u_guess, Δt = fill(Δt, N));
#     timestep=:Δt,
#     controls=:u,
#     initial=(x = x_init,)  # No final constraint
# )
# obj_soft = QuadraticRegularizer(:u, traj_soft, 1.0) +
#            TerminalObjective(x -> 100.0 * norm(x - x_goal)^2, :x, traj_soft)
# ```

# ## Next Steps

# - **Bilinear Control Tutorial**: Multiple drives and bounds
# - **Minimum Time Tutorial**: Optimize trajectory duration
# - **Smooth Controls Tutorial**: Add derivative penalties
