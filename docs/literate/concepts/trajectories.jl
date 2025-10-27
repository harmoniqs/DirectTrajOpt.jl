# # Trajectories

# ## What is a NamedTrajectory?

# A `NamedTrajectory` is the central data structure in DirectTrajOpt.jl. It stores:
# - **States** and **controls** over time
# - **Time step** information (fixed or variable)
# - **Boundary conditions** (initial, final, goal)
# - **Bounds** on variables
# - **Metadata** about which variables are controls, timesteps, etc.

using DirectTrajOpt
using NamedTrajectories

# ## Basic Construction

# ### Minimal Example

N = 10  # number of time steps

# Specify component data as a NamedTuple:
data = (
    x = randn(2, N),    # 2D state
    u = randn(1, N),    # 1D control
    Δt = fill(0.1, N)   # time step
)

traj = NamedTrajectory(
    data;
    timestep=:Δt,   # which variable represents time
    controls=:u     # which variable(s) are controls
)

# Access components:
println("State at time 1: ", traj.x[:, 1])
println("Control at time 5: ", traj.u[:, 5])
println("Total time: ", sum(traj.Δt))

# ## Trajectory Components

# ### States
# Variables that represent the system configuration. Can be multiple state vectors:

traj_multi = NamedTrajectory(
    (
        position = randn(3, N),  # 3D position
        velocity = randn(3, N),  # 3D velocity
        u = randn(2, N),
        Δt = fill(0.1, N)
    );
    timestep=:Δt,
    controls=:u
)

# ### Controls
# Variables that you can actuate. Can specify multiple control variables:

traj_multi_control = NamedTrajectory(
    (
        x = randn(2, N),
        u1 = randn(1, N),
        u2 = randn(1, N),
        Δt = fill(0.1, N)
    );
    timestep=:Δt,
    controls=(:u1, :u2)
)

# ### Time Steps

# **Fixed time**: All time steps equal (constant Δt)
traj_fixed_time = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,  # symbol pointing to timestep component
    controls=:u
)

# **Free time**: Each time step is a decision variable (with bounds)
traj_free_time = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=(:u, :Δt),  # Include Δt in controls for optimization
    bounds=(Δt = (0.01, 0.5),)  # Set bounds on time steps
)

# ## Boundary Conditions

# ### Initial Conditions

# Fix the starting state:
traj_initial = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0],)  # x₁ = [0, 0]
)

# Can also fix initial controls:
traj_initial_u = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0], u = [0.0])  # x₁ = [0, 0], u₁ = 0
)

# ### Final Conditions

# Fix the ending state:
traj_final = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    final=(x = [1.0, 0.0],)  # xₖ = [1, 0]
)

# ### Goal Conditions

# Similar to final, but typically used with terminal cost instead of hard constraint:
traj_goal = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    goal=(x = [1.0, 0.0],)  # target: xₖ → [1, 0]
)

# ### Complete Example

traj_complete = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0], u = [0.0]),
    final=(u = [0.0],),
    goal=(x = [1.0, 0.0],)
)

# ## Bounds on Variables

# Bounds constrain variables to lie within specified ranges.

# ### Scalar Bounds (Symmetric)

# A single number creates symmetric bounds: `-bound ≤ var ≤ bound`

traj_scalar_bound = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    bounds=(u = 1.0,)  # -1 ≤ u ≤ 1 for all components
)

# Applies to all components of the variable:
println("u bounds: ", traj_scalar_bound.bounds.u)
# Output: ([-1.0], [1.0])

# ### Tuple Bounds (Asymmetric)

# A tuple `(lower, upper)` creates asymmetric bounds:

traj_tuple_bound = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    bounds=(u = (-2.0, 1.0),)  # -2 ≤ u ≤ 1
)

println("u bounds: ", traj_tuple_bound.bounds.u)
# Output: ([-2.0], [1.0])

# ### Vector Bounds (Component-wise Symmetric)

# A vector creates component-specific symmetric bounds:

traj_vector_bound = NamedTrajectory(
    (x = randn(2, N), u = randn(2, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    bounds=(u = [1.0, 2.0],)  # -1 ≤ u₁ ≤ 1, -2 ≤ u₂ ≤ 2
)

println("u bounds: ", traj_vector_bound.bounds.u)
# Output: ([-1.0, -2.0], [1.0, 2.0])

# ### Tuple of Vectors (Component-wise Asymmetric)

# The most general form - specify lower and upper for each component:

traj_full_bound = NamedTrajectory(
    (x = randn(2, N), u = randn(2, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    bounds=(u = ([-2.0, -1.0], [1.0, 3.0]),)  # -2 ≤ u₁ ≤ 1, -1 ≤ u₂ ≤ 3
)

println("u bounds: ", traj_full_bound.bounds.u)
# Output: ([-2.0, -1.0], [1.0, 3.0])

# ### Multiple Variable Bounds

traj_multi_bounds = NamedTrajectory(
    (x = randn(2, N), u = randn(2, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    bounds=(
        x = 5.0,           # -5 ≤ x ≤ 5 (both components)
        u = [1.0, 2.0],    # component-specific
        Δt = (0.05, 0.15)  # 0.05 ≤ Δt ≤ 0.15
    )
)

# ### Time Step Bounds

# For free-time problems, bound the time steps:

traj_time_bounds = NamedTrajectory(
    (x = randn(2, N), u = randn(1, N), Δt = fill(0.1, N));
    timestep=:Δt,
    controls=:u,
    bounds=(
        u = 1.0,
        Δt = (0.01, 0.2)  # 0.01 ≤ Δt ≤ 0.2
    )
)

# ## Accessing Trajectory Data

# ### Direct Access
x_data = traj.x           # Get all states (2 × N matrix)
u_data = traj.u           # Get all controls (1 × N matrix)
x_final = traj.x[:, end]  # Get final state

# ### Metadata
println("Number of time steps: ", traj.N)
println("State dimension: ", traj.dims.x)
println("Control dimension: ", traj.dims.u)
println("Total dimension: ", traj.dim)

# ### Time Information
times = get_times(traj)  # Cumulative time at each knot point
total_time = sum(traj.Δt)

# ## Building Trajectories for Optimization

# ### Good Initialization Matters

# Start with a reasonable guess:
# - Linear interpolation between initial and final states
# - Zero or small random controls
# - Uniform time steps

x_init = [0.0, 0.0]
x_goal = [1.0, 1.0]

# Linear interpolation
x_guess = hcat([x_init + (x_goal - x_init) * (t / (N-1)) for t in 0:N-1]...)

traj_good_init = NamedTrajectory(
    (
        x = x_guess,
        u = zeros(1, N),
        Δt = fill(0.1, N)
    );
    timestep=:Δt,
    controls=:u,
    initial=(x = x_init,),
    final=(x = x_goal,)
)

# ## Common Patterns

# ### State Transfer Problem
# Drive from initial to final state with bounded controls

traj_transfer = NamedTrajectory(
    (x = randn(3, 50), u = randn(2, 50), Δt = fill(0.1, 50));
    timestep=:Δt,
    controls=:u,
    initial=(x = zeros(3),),
    final=(x = ones(3),),
    bounds=(u = 1.0,)
)

# ### Minimum Time Problem
# Free time steps, bounded, with time regularization

traj_mintime = NamedTrajectory(
    (x = randn(2, 30), u = randn(1, 30), Δt = fill(0.1, 30));
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0],),
    final=(x = [1.0, 0.0],),
    bounds=(
        u = 1.0,
        Δt = (0.01, 0.5)
    )
)

# ### Smooth Control Problem
# Include control derivatives for smoothness

traj_smooth = NamedTrajectory(
    (
        x = randn(2, 40),
        u = randn(2, 40),
        du = zeros(2, 40),   # control derivative
        Δt = fill(0.1, 40)
    );
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0], u = [0.0, 0.0]),
    final=(x = [1.0, 0.0], u = [0.0, 0.0]),
    bounds=(u = 1.0,)
)

# ## Summary

# | Concept | Syntax | Example |
# |---------|--------|---------|
# | Fixed time | `timestep=0.1` | `timestep=0.1` |
# | Free time | `timestep=:Δt` | `timestep=:Δt` |
# | Initial condition | `initial=(x = [...],)` | `initial=(x = [0, 0],)` |
# | Final condition | `final=(x = [...],)` | `final=(x = [1, 0],)` |
# | Scalar bound | `bounds=(u = 1.0,)` | `-1 ≤ u ≤ 1` |
# | Tuple bound | `bounds=(u = (-2, 1),)` | `-2 ≤ u ≤ 1` |
# | Vector bound | `bounds=(u = [1, 2],)` | `-1 ≤ u₁ ≤ 1, -2 ≤ u₂ ≤ 2` |
# | Full bound | `bounds=(u = ([-2,-1], [1,3]),)` | `-2 ≤ u₁ ≤ 1, -1 ≤ u₂ ≤ 3` |

# ## Next Steps

# - **Integrators**: Learn how dynamics are encoded
# - **Objectives**: Define cost functions on trajectories
# - **Tutorials**: See complete examples using trajectories
