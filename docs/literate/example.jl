# # Complete Example: Time-Optimal Bilinear Control

# This example demonstrates solving a time-optimal trajectory optimization problem with:
# - Multiple control inputs with bounds
# - Free time steps (variable Δt)
# - Combined objective (control effort + minimum time)

using DirectTrajOpt
using NamedTrajectories
using LinearAlgebra
using CairoMakie

# ## Problem Setup

# **System:** 3D oscillator with 2 control inputs
# ```math
# \dot{x} = (G_0 + u_1 G_1 + u_2 G_2) x
# ```

# **Goal:** Drive from `[1, 0, 0]` to `[0, 0, 1]` minimizing `∫ ||u||² dt + w·T`

# **Constraints:** `-1 ≤ u ≤ 1`, `0.05 ≤ Δt ≤ 0.3`

# ## Define System Dynamics

G_drift = [
    0.0 1.0 0.0;
    -1.0 0.0 0.0;
    0.0 0.0 -0.1
]

G_drives = [
    [
        1.0 0.0 0.0;
        0.0 0.0 0.0;
        0.0 0.0 0.0
    ],
    [
        0.0 0.0 0.0;
        0.0 0.0 1.0;
        0.0 1.0 0.0
    ],
]

G = u -> G_drift + sum(u .* G_drives)

# ## Create Trajectory

N = 50
x_init = [1.0, 0.0, 0.0]
x_goal = [0.0, 0.0, 1.0]
x_guess = hcat([x_init + (x_goal - x_init) * (k/(N-1)) for k = 0:(N-1)]...)

traj = NamedTrajectory(
    (x = x_guess, u = 0.1 * randn(2, N), Δt = fill(0.15, N));
    timestep = :Δt,
    controls = (:u, :Δt),
    initial = (x = x_init,),
    final = (x = x_goal,),
    bounds = (u = 1.0, Δt = (0.05, 0.3)),
)

# ## Build and Solve Problem

integrator = BilinearIntegrator(G, :x, :u, traj)

obj = (QuadraticRegularizer(:u, traj, 1.0) + 0.5 * MinimumTimeObjective(traj, 1.0))

prob = DirectTrajOptProblem(traj, obj, integrator)

prob

#-

solve!(prob; max_iter = 50)

# ## Visualize Solution

plot(prob.trajectory) # See NamedTrajectories.jl documentation for plotting options

# ## Analyze Solution

x_sol = prob.trajectory.x
u_sol = prob.trajectory.u
Δt_sol = prob.trajectory.Δt

println("Solution found!")
println("  Total time: $(sum(Δt_sol)) seconds")
println("  Δt range: [$(minimum(Δt_sol)), $(maximum(Δt_sol))]")
println("  Max |u₁|: $(maximum(abs.(u_sol[1,:])))")
println("  Max |u₂|: $(maximum(abs.(u_sol[2,:])))")
println("  Final error: $(norm(x_sol[:,end] - x_goal))")

# ## Key Insights

# **Free time optimization**: Variable Δt allows the optimizer to adjust trajectory speed,
# with shorter steps where control is needed and longer steps in smooth regions.

# **Control bounds**: With time weight 0.5, controls don't fully saturate. Increase the
# weight to push toward bang-bang control.

# **Combined objectives**: The `+` operator makes it easy to balance multiple goals.

# ## Exercises

# **1. Bang-bang control:** Set time weight to 5.0 - do controls saturate the bounds?

# **2. Fixed time:** Remove `Δt` from controls and compare total time.

# **3. Add waypoint:** Require passing through `[0.5, 0, 0.5]` at the midpoint:
# ```julia
# constraint = NonlinearKnotPointConstraint(
#     x -> x - [0.5, 0, 0.5], :x, traj;
#     times=[div(N,2)], equality=true
# )
# prob = DirectTrajOptProblem(traj, obj, integrator; constraints=[constraint])
# ```

# **4. Different goal:** Try reaching `[0, 1, 0]` or `[0.5, 0.5, 0.5]`

# **5. Tighter bounds:** Use `bounds=(u = 0.5, Δt = (0.05, 0.3))` - how does time change?
