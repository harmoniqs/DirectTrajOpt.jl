# DirectTrajOpt.jl Context

> AI-friendly context for maintaining consistency. Update this when making significant changes.

## Package Purpose

DirectTrajOpt.jl provides a **general-purpose direct trajectory optimization framework** using nonlinear programming. It is **not quantum-specific** - it solves the generic problem:

```
minimize    J(z)                    # Objective
subject to  f(z_k+1, z_k) = 0      # Dynamics (integrators)
            c(z) ≥ 0               # Constraints
            bounds on z            # Variable bounds
```

where `z` is the flattened trajectory vector. The underlying solver is **Ipopt**.

## Core Abstractions

### NamedTrajectory (from NamedTrajectories.jl)

The data structure holding the optimization variables. DirectTrajOpt operates on flattened vectors but users interact via named components:

```julia
traj = NamedTrajectory(
    (x = rand(2, T), u = rand(1, T), Δt = fill(0.1, T));
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0],),
    final=(x = [1.0, 0.0],),
    bounds=(u = (-1.0, 1.0),)
)

# Access
traj.x                # Matrix (2×T)
traj[k]              # KnotPoint at timestep k
traj.datavec         # Flattened vector for optimizer
```

### AbstractIntegrator

Enforces dynamics constraints `f(z_{k+1}, z_k, u_k, ...) = 0`. Each integrator:
- Has a dimension (`dim`) - number of constraint equations per timestep
- Implements sparse Jacobian/Hessian computation
- Operates on trajectory components by name

**Built-in integrators:**

| Type | Purpose |
|------|---------|
| `BilinearIntegrator` | Matrix exponential for `ẋ = (A + Σ uᵢ Bᵢ) x` |
| `DerivativeIntegrator` | Smoothness via `u_{k+1} - u_k - Δt * du_k ≈ 0` |
| `TimeDependentBilinearIntegrator` | Time-varying drift/drives |

**Interface:**
```julia
# Evaluate constraint residual
evaluate!(δ, integrator, traj)  # δ should be ≈ 0

# Sparse derivatives (filled in-place)
jacobian_structure(integrator)  # Returns sparse pattern
jacobian!(∂f, integrator, z₁, z₂, k)  # Fill values

# Hessian of Lagrangian (weighted by multipliers μ)
hessian_of_lagrangian(integrator, traj, μ)
```

### Objective

Cost functions with automatic differentiation support:

```julia
# Built-in objectives
QuadraticRegularizer(:u, traj, R)       # ½ R Σ ||u_k||²
QuadraticObjective(:x, traj, Q, x_ref)  # ½ Q Σ ||x_k - x_ref||²
MinimumTimeObjective(traj)              # Σ Δt_k

# Objectives compose via +, *
total_obj = Q_obj + 0.1 * R_obj + mintime_obj

# Structure
obj = Objective(L, ∇L, ∂²L, ∂²L_structure)
# L(Z⃗) → scalar cost
# ∇L(Z⃗) → gradient vector
# ∂²L(Z⃗) → Hessian values (sparse)
# ∂²L_structure → Vector{Tuple{Int,Int}} of (i,j) indices
```

### AbstractConstraint

Path and boundary constraints:

```julia
# Linear constraints (from trajectory bounds/initial/final)
EqualityConstraint(traj, indices, values)
BoundsConstraint(traj, :u, (lower, upper))

# Nonlinear constraints
NonlinearKnotPointConstraint(g, ∂g, ∂²g, indices)  # Per timestep
NonlinearGlobalConstraint(g, ∂g, ∂²g)              # Over trajectory

# Auto-extract from trajectory
constraints = get_trajectory_constraints(traj)
```

### TrajectoryDynamics

Aggregates multiple integrators into full dynamics:

```julia
dynamics = TrajectoryDynamics([
    BilinearIntegrator(G, traj, :x, :u),
    DerivativeIntegrator(traj, :u, :du),
    DerivativeIntegrator(traj, :du, :ddu)
], traj)

# Uses Threads.@threads for parallel evaluation across timesteps
```

### DirectTrajOptProblem

Main container that combines everything:

```julia
prob = DirectTrajOptProblem(
    traj,           # NamedTrajectory
    obj,            # Objective
    integrators;    # Vector{AbstractIntegrator} or TrajectoryDynamics
    constraints=[   # Optional additional constraints
        my_constraint1,
        my_constraint2
    ]
)

# Solve
solve!(prob; 
    max_iter=100,
    tol=1e-6,
    verbose=true,
    print_level=5
)

# Solution is written back to prob.trajectory
```

## Slice Indexing

Use `slice(k, comps, dim)` from TrajectoryIndexingUtils to index into flattened vectors:

```julia
using TrajectoryIndexingUtils: slice

Z⃗ = traj.datavec
dim = traj.dim  # Total variables per timestep

# Get all variables at timestep k
zₖ = Z⃗[slice(k, dim)]

# Get specific components at timestep k
x_comps = traj.components[:x]  # e.g., 1:2
xₖ = Z⃗[slice(k, x_comps, dim)]
```

## Sparse Derivative Computation

All derivatives use pre-computed sparsity patterns:

```julia
# 1. Get structure (which entries are non-zero)
∂f = jacobian_structure(integrator)  # Sparse matrix

# 2. Fill values (repeated during optimization)
jacobian!(∂f, integrator, z₁, z₂, k)

# Hessian returns (values, structure)
vals, structure = hessian_of_lagrangian(integrator, traj, μ)
# structure is Vector{Tuple{Int,Int}} of (row, col) indices
```

## Testing Integrators

Use finite difference validation:

```julia
using DirectTrajOpt: test_integrator

integrator = BilinearIntegrator(G, traj, :x, :u)
test_integrator(integrator)  # Validates Jacobian/Hessian vs finite diff
```

## Package Dependencies

- **Ipopt.jl** - Interior point optimizer (the actual solver)
- **ForwardDiff.jl** - Automatic differentiation for objectives
- **FiniteDiff.jl** - Finite differences for testing
- **NamedTrajectories.jl** - Trajectory data structure
- **TrajectoryIndexingUtils.jl** - `slice` function for indexing

## Module Organization

```
DirectTrajOpt.jl/src/
├── DirectTrajOpt.jl      # Main module, exports
├── common_interface.jl   # Shared abstract types
├── problems.jl           # DirectTrajOptProblem
├── solvers.jl            # solve! implementation
├── integrators/
│   ├── _integrators.jl   # Module definition
│   ├── bilinear.jl       # BilinearIntegrator
│   └── derivative.jl     # DerivativeIntegrator
├── objectives/
│   ├── _objectives.jl    # Module definition
│   ├── regularizers.jl   # QuadraticRegularizer
│   └── ...
├── constraints/
│   ├── _constraints.jl   # Module definition
│   ├── linear.jl         # EqualityConstraint, BoundsConstraint
│   └── nonlinear.jl      # NonlinearKnotPointConstraint
└── solvers/
    └── ipopt.jl          # Ipopt interface
```

## Common Patterns

### Creating a Custom Integrator

```julia
struct MyIntegrator <: AbstractIntegrator
    dim::Int
    # ... fields ...
end

function (integrator::MyIntegrator)(δ, z₁, z₂, k)
    # Fill δ with constraint residual
    # δ should equal 0 when constraints satisfied
end

function DirectTrajOpt.jacobian_structure(integrator::MyIntegrator)
    # Return sparse matrix with pattern
end

function DirectTrajOpt.jacobian!(∂f, integrator::MyIntegrator, z₁, z₂, k)
    # Fill ∂f with actual values
end

function DirectTrajOpt.hessian_of_lagrangian(integrator::MyIntegrator, traj, μ)
    # Return (values, structure) for Hessian
end
```

### Creating a Custom Objective

```julia
function MyObjective(traj::NamedTrajectory, weight::Float64)
    function L(Z⃗)
        # Compute cost
    end
    
    function ∇L(Z⃗)
        # Compute gradient (use ForwardDiff if needed)
        ForwardDiff.gradient(L, Z⃗)
    end
    
    # For Hessian, return values and structure
    function ∂²L(Z⃗)
        H = ForwardDiff.hessian(L, Z⃗)
        # Extract non-zero values
    end
    
    function ∂²L_structure()
        # Return Vector{Tuple{Int,Int}} of (i,j) indices
    end
    
    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end
```

## Gotchas

- Always use `slice()` for trajectory indexing - never manual arithmetic
- Integrators operate in-place: first argument `δ` is modified
- Hessian functions take Lagrange multipliers `μ` and return weighted Hessian
- Thread safety: dynamics functions use `Threads.@threads` over timesteps
- Sparsity patterns must be exact - wrong structure causes silent failures
- The trajectory in `prob.trajectory` is mutated during solve
