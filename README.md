# DirectTrajOpt.jl

<!--```@raw html-->
<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>Documentation</b>
        <br>
        <a href="https://docs.harmoniqs.co/DirectTrajOpt/stable/">
          <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable"/>
        </a>
        <a href="https://docs.harmoniqs.co/DirectTrajOpt/dev/">
          <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev"/>
        </a>
      </td>
      <td align="center">
        <b>Build Status</b>
        <br>
        <a href="https://github.com/harmoniqs/DirectTrajOpt.jl/actions/workflows/CI.yml?query=branch%3Amain">
          <img src="https://github.com/harmoniqs/DirectTrajOpt.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status"/>
        </a>
        <a href="https://codecov.io/gh/harmoniqs/DirectTrajOpt.jl">
          <img src="https://codecov.io/gh/harmoniqs/DirectTrajOpt.jl/branch/main/graph/badge.svg" alt="Coverage"/>
        </a>
      </td>
      <td align="center">
        <b>License</b>
        <br>
        <a href="https://opensource.org/licenses/MIT">
          <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
        </a>
      </td>
    </tr>
  </table>
</div>
<!--```-->

**DirectTrajOpt.jl** is a framework for direct trajectory optimization via nonlinear programming. It converts continuous optimal control problems into finite-dimensional NLPs using direct transcription, then solves them with [Ipopt](https://github.com/jump-dev/Ipopt.jl).

## Problem Formulation

DirectTrajOpt solves problems of the form:

```math
\begin{align*}
\underset{x_{1:N}, u_{1:N}}{\text{minimize}} \quad & J(x_{1:N}, u_{1:N}) \\
\text{subject to} \quad & f(x_{k+1}, x_k, u_k, \Delta t, t_k) = 0, \quad k = 1, \ldots, N-1\\
& c_k(x_k, u_k) \geq 0, \quad k = 1, \ldots, N \\
& x_1 = x_{\text{init}}, \quad x_N = x_{\text{goal}} \\
\end{align*}
```

where:
- `J(x, u)` is the objective function to minimize
- `f(.)` represents system dynamics encoded via *integrators*
- `c(.)` represents additional nonlinear constraints
- `x` is the state trajectory
- `u` is the control trajectory

## Installation

```julia
using Pkg
Pkg.add("DirectTrajOpt")
```

## Quick Example

```julia
using DirectTrajOpt
using NamedTrajectories

# Define trajectory
traj = NamedTrajectory(
    (x = randn(2, 50), u = randn(1, 50), Δt = fill(0.1, 50));
    timestep=:Δt,
    controls=:u,
    initial=(x = [0.0, 0.0],),
    final=(x = [1.0, 0.0],)
)

# Define dynamics: dx/dt = (A + u * B) * x
G_drift = [-0.1 1.0; -1.0 -0.1]
G_drives = [[0.0 1.0; 1.0 0.0]]
G = u -> G_drift + sum(u .* G_drives)
integrator = BilinearIntegrator(G, :x, :u, traj)

# Define objective
obj = QuadraticRegularizer(:u, traj, 1.0)

# Create and solve problem
prob = DirectTrajOptProblem(traj, obj, integrator)
solve!(prob; max_iter=100)
```

## Key Features

- **Flexible dynamics**: Define system evolution via bilinear, time-dependent, or derivative integrators
- **Modular objectives**: Combine cost terms with `+` and `*` (regularization, minimum time, terminal cost, etc.)
- **Constraint support**: Bounds, equality, nonlinear, symmetry, and L1 slack constraints
- **Automatic differentiation**: Sparse Jacobians and Hessians via ForwardDiff
- **Solver callbacks**: Monitor and control the optimization process

## Contributing

### Building Documentation

This package uses a shared Documenter config. First-time setup:
```bash
./docs/get_docs_utils.sh
```

Build docs:
```bash
julia --project=docs docs/make.jl
```

Live editing:
```bash
julia --project=docs -e '
  using LiveServer, DirectTrajOpt, Revise
  servedocs(
    literate_dir="docs/literate",
    skip_dirs=["docs/src/generated", "docs/src/assets/"],
    skip_files=["docs/src/index.md"]
  )'
```

> **Note:** `servedocs` will loop if it watches generated files. Ensure all generated files are in the skip dirs/files args.
