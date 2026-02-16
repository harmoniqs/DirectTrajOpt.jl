# # Tutorial: Bilinear Control with Multiple Drives

# This tutorial demonstrates a more complex bilinear control problem with:
# - Multiple control inputs
# - Control bounds
# - A 3D state space

# ## Problem Description

# Control a 3D system with 2 independent control inputs.

# **Dynamics:**
# ```math
# \dot{x} = (G_0 + u_1 G_1 + u_2 G_2) x
# ```

# **Goal:** Navigate from `[1, 0, 0]` to `[0, 0, 1]` with bounded controls

using DirectTrajOpt
using NamedTrajectories
using LinearAlgebra
using Statistics
using Printf

# ## Step 1: Define the System

# ### 3D System with Two Drives

# Drift term - natural evolution
G_drift = [
     0.0   1.0   0.0;
    -1.0   0.0   0.0;
     0.0   0.0  -0.1
]

# Drive terms - control influences
G_drive_1 = [
    1.0  0.0  0.0;
    0.0  0.0  0.0;
    0.0  0.0  0.0
]  # Controls first state

G_drive_2 = [
    0.0  0.0  0.0;
    0.0  0.0  1.0;
    0.0  1.0  0.0
]  # Controls coupling between states 2 and 3

G_drives = [G_drive_1, G_drive_2]

# Generator function
G = u -> G_drift + sum(u .* G_drives)

println("System dynamics:")
println("  State dimension: 3")
println("  Number of controls: 2")
println("  G_drift creates oscillation in x₁-x₂ plane with decay in x₃")
println("  u₁ drives x₁ component")
println("  u₂ couples x₂ and x₃")

# ## Step 2: Set Up the Problem

# ### Time Discretization

N = 60
Δt = 0.15
total_time = N * Δt

println("\nTime discretization:")
println("  Time steps: $N")
println("  Step size: $Δt")
println("  Total time: $total_time seconds")

# ### Boundary Conditions

x_init = [1.0, 0.0, 0.0]
x_goal = [0.0, 0.0, 1.0]

println("\nBoundary conditions:")
println("  Initial state: $x_init")
println("  Goal state: $x_goal")

# ### Initial Guess

# Linear interpolation for states
x_guess = hcat([x_init + (x_goal - x_init) * (t/(N-1)) for t in 0:N-1]...)

# Small random controls
u_guess = 0.1 * randn(2, N)

# ## Step 3: Create Trajectory with Bounds

# Control bounds: -1 ≤ u ≤ 1 for both controls

traj = NamedTrajectory(
    (
        x = x_guess,
        u = u_guess,
        Δt = fill(Δt, N)
    );
    timestep=:Δt,
    controls=:u,
    initial=(x = x_init,),
    final=(x = x_goal,),
    bounds=(u = 1.0,)  # -1 ≤ u ≤ 1
)

println("\nTrajectory created:")
println("  Control bounds: ", traj.bounds.u)

# ## Step 4: Define Dynamics and Objective

# Dynamics integrator
integrator = BilinearIntegrator(G, :x, :u, traj)

# Objective: minimize control effort
R_u = 1.0  # control weight
obj = QuadraticRegularizer(:u, traj, R_u)

println("\nObjective: minimize ∫ ||u||² dt")
println("  Control weight: $R_u")

# ## Step 5: Solve the Problem

prob = DirectTrajOptProblem(traj, obj, integrator)

println("\n" * "="^50)
println("Solving optimization problem...")
println("="^50)

solve!(prob; max_iter=150, verbose=false)

println("="^50)
println("Optimization complete!")
println("="^50)

# ## Step 6: Analyze the Solution

x_sol = prob.trajectory.x
u_sol = prob.trajectory.u
times = cumsum([0.0; prob.trajectory.Δt[:]])

# ### Goal Reaching

println("\nGoal reaching:")
println("  Initial state: ", x_sol[:, 1])
println("  Final state:   ", x_sol[:, end])
println("  Goal state:    ", x_goal)
println("  Final error:   ", norm(x_sol[:, end] - x_goal))

# ### Control Statistics

println("\nControl statistics:")
for i in 1:2
    u_i = u_sol[i, :]
    println("  u$i:")
    println("    Max magnitude: ", maximum(abs.(u_i)))
    println("    Mean magnitude: ", mean(abs.(u_i)))
    println("    Total norm: ", norm(u_i))
    
    # Check bound satisfaction
    if all(-1.0 .<= u_i .<= 1.0)
        println("    ✓ Bounds satisfied")
    else
        println("    ✗ Bounds violated!")
    end
end

# ### State Trajectory Analysis

println("\nState trajectory:")
println("  Max |x₁|: ", maximum(abs.(x_sol[1, :])))
println("  Max |x₂|: ", maximum(abs.(x_sol[2, :])))
println("  Max |x₃|: ", maximum(abs.(x_sol[3, :])))

# ## Step 7: Detailed Results

println("\n" * "="^50)
println("SOLUTION DETAILS")
println("="^50)

println("\nState trajectory (selected time points):")
println("Time  |   x₁    |   x₂    |   x₃")
println("-"^40)
for k in [1, 10, 20, 30, 40, 50, N]
    t = times[k]
    println(@sprintf("%.2f | %7.4f | %7.4f | %7.4f", 
        t, x_sol[1,k], x_sol[2,k], x_sol[3,k]))
end

println("\nControl trajectory (selected time points):")
println("Time  |   u₁    |   u₂")
println("-"^30)
for k in [1, 10, 20, 30, 40, 50, N]
    t = times[k]
    println(@sprintf("%.2f | %7.4f | %7.4f", 
        t, u_sol[1,k], u_sol[2,k]))
end

# ## Step 8: Verify Dynamics Satisfaction

println("\nDynamics verification:")
max_error = 0.0
for k in 1:N-1
    x_k = x_sol[:, k]
    u_k = u_sol[:, k]
    Δt_k = prob.trajectory.Δt[k]
    
    # Predicted next state
    x_k1_pred = exp(Δt_k * G(u_k)) * x_k
    x_k1_actual = x_sol[:, k+1]
    
    error = norm(x_k1_pred - x_k1_actual)
    max_error = max(max_error, error)
end
println("  Maximum dynamics error: $max_error")

# ## Comparison: Different Control Weights

println("\n" * "="^50)
println("EXPLORING DIFFERENT CONTROL WEIGHTS")
println("="^50)

# Try with higher control penalty
traj_high = NamedTrajectory(
    (x = x_guess, u = 0.1*randn(2,N), Δt = fill(Δt, N));
    timestep=:Δt, controls=:u,
    initial=(x = x_init,), final=(x = x_goal,),
    bounds=(u = 1.0,)
)

obj_high = QuadraticRegularizer(:u, traj_high, 10.0)  # 10x larger weight
integrator_high = BilinearIntegrator(G, :x, :u, traj_high)
prob_high = DirectTrajOptProblem(traj_high, obj_high, integrator_high)

println("\nSolving with high control weight (R=10.0)...")
solve!(prob_high; max_iter=150, verbose=false)

u_sol_high = prob_high.trajectory.u

println("\nComparison:")
println("  Original (R=1.0):")
println("    ||u||: ", norm(u_sol))
println("  High weight (R=10.0):")
println("    ||u||: ", norm(u_sol_high))
println("  Control reduction: ", (1 - norm(u_sol_high)/norm(u_sol)) * 100, "%")

# ## Key Observations

println("\n" * "="^50)
println("KEY TAKEAWAYS")
println("="^50)

println("""
1. **Multiple controls** allow independent actuation of different system modes

2. **Bounds are strictly enforced** - check that max|u| ≤ 1

3. **Control weight** affects aggressiveness:
   - Lower weight → larger controls, potentially saturating bounds
   - Higher weight → gentler controls, stays away from bounds

4. **Initial guess matters** for bounded problems:
   - Random guess works for this problem
   - More complex problems may need better initialization

5. **BilinearIntegrator** handles multi-input systems naturally:
   - Just provide all drive matrices in a vector
   - Generator sums u₁G₁ + u₂G₂ + ...
""")

# ## Exercises

println("\n" * "="^50)
println("EXERCISES")
println("="^50)

println("""
Try these modifications:

### Exercise 1: Asymmetric Bounds
Set different bounds for each control:
```julia
bounds=(u = ([-1.0, -0.5], [1.0, 2.0]),)
```

### Exercise 2: Initial Control Constraints
Start and end with zero control:
```julia
initial=(x = x_init, u = [0.0, 0.0]),
final=(x = x_goal, u = [0.0, 0.0])
```

### Exercise 3: Intermediate Waypoint
Add a waypoint constraint at t=30:
```julia
waypoint = [0.5, 0.5, 0.5]
constraint = NonlinearKnotPointConstraint(
    x -> x - waypoint, :x, traj;
    times=[30], equality=true
)
# Add to problem with: constraints=[constraint]
```

### Exercise 4: Different Goal
Try reaching [0, 1, 0] instead of [0, 0, 1]

### Exercise 5: Add Minimum Time
Make Δt variable and add MinimumTimeObjective:
```julia
obj = QuadraticRegularizer(:u, traj, 1.0) + 
      MinimumTimeObjective(traj, 0.5)
bounds=(u = 1.0, Δt = (0.05, 0.3))
```
""")

# ## Next Steps

println("""
- **Minimum Time Tutorial**: Optimize trajectory duration
- **Smooth Controls Tutorial**: Add derivative penalties for implementability
- **Custom Constraints Tutorial**: Add obstacles and complex path constraints
""")
