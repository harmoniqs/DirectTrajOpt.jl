# # Tutorial: Minimum Time Problems

# This tutorial shows how to solve **time-optimal** problems where trajectory duration
# is minimized alongside other objectives.

# ## Problem Description

# Find the **fastest trajectory** from start to goal with bounded controls.

# **Dynamics:**
# ```math
# \dot{x} = (G_0 + u_1 G_1) x
# ```

# **Objective:** Minimize total time + control effort

# **Constraints:** |u| ≤ 1

using DirectTrajOpt
using NamedTrajectories
using LinearAlgebra
using Statistics
using Printf

# ## Fixed Time vs Free Time

println("="^50)
println("MINIMUM TIME TRAJECTORY OPTIMIZATION")
println("="^50)

println("""
Two approaches:
1. **Fixed time**: All Δt equal and constant
2. **Free time**: Each Δt is a variable (what we'll use)

Free time allows the optimizer to adjust trajectory duration.
""")

# ## Step 1: System Definition

G_drift = [
    -0.1 1.0;
    -1.0 -0.1
]

G_drives = [[
    0.0 1.0;
    1.0 0.0
]]

G = u -> G_drift + sum(u .* G_drives)

println("System: 2D damped oscillator with symmetric control coupling")

# ## Step 2: Trajectory Setup

N = 40# Number of time steps (fewer than before)
Δt_init = 0.15  # Initial guess for time step

x_init = [0.0, 0.0]
x_goal = [1.0, 0.0]

# Initial guess
x_guess = hcat([x_init + (x_goal - x_init) * (t/(N-1)) for t = 0:(N-1)]...)
u_guess = 0.1 * randn(1, N)
Δt_guess = fill(Δt_init, N)

println("\nProblem setup:")
println("  Time steps: $N")
println("  Initial guess for Δt: $Δt_init")
println("  Initial total time: ", sum(Δt_guess))

# ## Step 3: Create Free-Time Trajectory

# Key: timestep=:Δt makes time steps decision variables

traj_mintime = NamedTrajectory(
    (x = x_guess, u = u_guess, Δt = Δt_guess);
    timestep = :Δt,  # Time is a variable!
    controls = :u,
    initial = (x = x_init,),
    final = (x = x_goal,),
    bounds = (
        u = 1.0,            # -1 ≤ u ≤ 1
        Δt = (0.01, 0.5),    # 0.01 ≤ Δt ≤ 0.5
    ),
)

println("\nTrajectory bounds:")
println("  Control: ", traj_mintime.bounds.u)
println("  Time step: ", traj_mintime.bounds.Δt)

# ## Step 4: Define Objectives

# ### Time Minimization Weight

# The key parameter: balance speed vs control effort

w_time = 1.0   # Weight on total time
w_control = 1e-2  # Weight on control effort

obj_mintime = (
    w_control * QuadraticRegularizer(:u, traj_mintime, 1.0) +
    w_time * MinimumTimeObjective(traj_mintime, 1.0)
)

println("\nObjective weights:")
println("  Control effort: $w_control")
println("  Time: $w_time")
println("  → Emphasizes minimizing time")

# ## Step 5: Solve Minimum Time Problem

integrator_mintime = BilinearIntegrator(G, :x, :u, traj_mintime)
prob_mintime = DirectTrajOptProblem(traj_mintime, obj_mintime, integrator_mintime)

prob_mintime

#-

println("Solving minimum time problem...")
println("="^50)

solve!(prob_mintime; max_iter = 200, verbose = false)

println("="^50)
println("Minimum time solution found!")
println("="^50)

# ## Step 6: Analyze Time-Optimal Solution

x_sol_mintime = prob_mintime.trajectory.x
u_sol_mintime = prob_mintime.trajectory.u
Δt_sol_mintime = prob_mintime.trajectory.Δt

total_time_mintime = sum(Δt_sol_mintime)

println("\nMinimum time solution:")
println("  Total time: $total_time_mintime seconds")
println("  Average Δt: ", mean(Δt_sol_mintime))
println("  Min Δt: ", minimum(Δt_sol_mintime))
println("  Max Δt: ", maximum(Δt_sol_mintime))

println("\nControl statistics:")
println("  Max |u|: ", maximum(abs.(u_sol_mintime)))
println("  Mean |u|: ", mean(abs.(u_sol_mintime)))
println("  ||u||: ", norm(u_sol_mintime))

# Check if controls saturate
u_saturated = sum(abs.(u_sol_mintime) .> 0.99)
println("  Time steps with |u| > 0.99: $u_saturated / $N")

# ## Step 7: Comparison with Fixed-Time Solution

println("\n" * "="^50)
println("COMPARISON: MINIMUM TIME vs FIXED TIME")
println("="^50)

# Solve fixed-time problem with same total time
Δt_fixed = total_time_mintime / N

traj_fixed = NamedTrajectory(
    (x = x_guess, u = u_guess, Δt = fill(Δt_fixed, N));
    timestep = :Δt,
    controls = :u,
    initial = (x = x_init,),
    final = (x = x_goal,),
    bounds = (u = 1.0,),
)

obj_fixed = QuadraticRegularizer(:u, traj_fixed, 1.0)
integrator_fixed = BilinearIntegrator(G, :x, :u, traj_fixed)
prob_fixed = DirectTrajOptProblem(traj_fixed, obj_fixed, integrator_fixed)

println("\nSolving fixed-time problem with T = $total_time_mintime seconds...")
solve!(prob_fixed; max_iter = 150, verbose = false)

u_sol_fixed = prob_fixed.trajectory.u

println("\nComparison:")
println("  Minimum time:")
println("    Total time: $total_time_mintime s")
println("    ||u||: ", norm(u_sol_mintime))
println("    Max |u|: ", maximum(abs.(u_sol_mintime)))
println("  Fixed time:")
println("    Total time: ", sum(prob_fixed.trajectory.Δt), " s")
println("    ||u||: ", norm(u_sol_fixed))
println("    Max |u|: ", maximum(abs.(u_sol_fixed)))

# ## Step 8: Effect of Time Weight

println("\n" * "="^50)
println("EXPLORING TIME WEIGHT EFFECTS")
println("="^50)

# Try different time weights
time_weights = [0.1, 1.0, 10.0]
results = []

for w_t in time_weights
    traj_test = NamedTrajectory(
        (x = x_guess, u = u_guess, Δt = Δt_guess);
        timestep = :Δt,
        controls = :u,
        initial = (x = x_init,),
        final = (x = x_goal,),
        bounds = (u = 1.0, Δt = (0.01, 0.5)),
    )

    obj_test = (
        1e-2 * QuadraticRegularizer(:u, traj_test, 1.0) +
        w_t * MinimumTimeObjective(traj_test, 1.0)
    )

    integrator_test = BilinearIntegrator(G, :x, :u, traj_test)
    prob_test = DirectTrajOptProblem(traj_test, obj_test, integrator_test)

    solve!(prob_test; max_iter = 200, verbose = false)

    push!(
        results,
        (
            weight = w_t,
            time = sum(prob_test.trajectory.Δt),
            control_norm = norm(prob_test.trajectory.u),
            max_control = maximum(abs.(prob_test.trajectory.u)),
        ),
    )
end

println("\nTime weight effects:")
println("Weight | Total Time | ||u||   | Max |u|")
println("-"^45)
for r in results
    println(
        @sprintf(
            "%.1f   | %.4f s   | %.4f | %.4f",
            r.weight,
            r.time,
            r.control_norm,
            r.max_control
        )
    )
end

println("\nObservations:")
println("  - Lower weight → slower trajectory, gentler controls")
println("  - Higher weight → faster trajectory, more aggressive controls")

# ## Step 9: Time Step Adaptation

Δt_variation = std(Δt_sol_mintime)
println("\nTime step adaptation:")
println("  Std dev(Δt): ", Δt_variation)
println("  Coefficient of variation: ", @sprintf("%.3f", Δt_variation / mean(Δt_sol_mintime)))

# ## Key Insights
#
# 1. **Free time variables**: Setting `timestep=:Δt` makes time steps optimizable
# 2. **Time bounds are crucial**: Lower bound prevents Δt -> 0, upper bound prevents unrealistically large steps
# 3. **Time weight balances speed vs control**: High weight -> fast but aggressive, low weight -> slow but gentle
# 4. **Control saturation**: Time-optimal solutions often saturate control bounds (bang-bang behavior)
# 5. **Non-uniform time steps**: Optimizer may choose variable Δt — larger steps where less control is needed
# 6. **Initial guess**: Start with reasonable Δt to help convergence

# ## Best Practices
#
# ### Time Step Bounds
# - **Lower bound**: ~0.01 to 0.05 (prevent numerical issues)
# - **Upper bound**: 1/10 to 1/5 of expected total time
# - Start conservative, relax if needed
#
# ### Control Weights
# - Usually small (1e-3 to 1e-2) for regularization
# - Just enough to ensure well-conditioned problem
# - Too large defeats the purpose of time minimization
#
# ### Time Weights
# - Start with ~1.0 and adjust
# - Increase to prioritize speed more
# - Decrease if controls become too aggressive
#
# ### Number of Time Steps
# - Fewer steps = less resolution, harder to satisfy dynamics
# - More steps = more variables, slower solve
# - Rule of thumb: 30-100 steps for most problems
#
# ### Initialization
# - Use solution from fixed-time problem as warm start
# - Or solve with high control weight first, then reduce

# ## Exercises
#
# ### Exercise 1: Bang-Bang Control
# Increase time weight to `w_time=100.0`. Do controls saturate more?
#
# ### Exercise 2: Time Step Constraints
# Try tighter bounds: `Δt ∈ [0.05, 0.15]`. How does total time change?
#
# ### Exercise 3: Longer Distance
# Change goal to `x_goal = [2.0, 0.0]`. How does optimal time scale?
#
# ### Exercise 4: Multiple Objectives
# Add terminal cost with soft goal:
# ```julia
# obj = w_control * QuadraticRegularizer(:u, traj, 1.0) +
#       w_time * MinimumTimeObjective(traj, 1.0) +
#       100.0 * TerminalObjective(x -> norm(x - x_goal)^2, :x, traj)
# ```
#
# ### Exercise 5: Warm Starting
# Solve fixed-time problem first, use as initial guess for free-time:
# ```julia
# traj_warm = NamedTrajectory(
#     (x = prob_fixed.trajectory.x,
#      u = prob_fixed.trajectory.u,
#      Δt = Δt_guess);
#     # ... rest of setup
# )
# ```

# ## Next Steps
#
# - **Smooth Controls Tutorial**: Add derivative penalties while minimizing time
# - **How-To Guide: Tune the Solver**: Improve convergence for difficult problems
# - **Advanced Topics: Performance**: Optimize large-scale problems
