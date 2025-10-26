# DirectTrajOpt.jl


<!--```@raw html-->
<!-- <div align="center">
  <a href="https://github.com/harmoniqs/Piccolo.jl">
    <img src="assets/logo.svg" alt="Piccolo.jl" width="25%"/>
  </a>
</div> -->

<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>Documentation</b>
        <br>
        <a href="https://docs.harmoniqs.co/DirectTrajOpt/dev/">
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
    </tr>
  </table>
</div>

<div align="center">
<br>
</div>
<!--```-->

**DirectTrajOpt.jl** provides a framework for setting up and solving direct trajectory optimization problems using nonlinear programming.

## Problem Formulation

DirectTrajOpt solves problems of the form:

```math
\begin{align*}
\underset{x_{1:T}, u_{1:T}}{\text{minimize}} \quad & J(x_{1:T}, u_{1:T}) \\
\text{subject to} \quad & f(x_{k+1}, x_k, u_k, \Delta t, t_k) = 0, \quad k = 1, \ldots, T-1\\
& c_k(x_k, u_k) \geq 0, \quad k = 1, \ldots, T \\
& x_1 = x_{\text{init}}, \quad x_T = x_{\text{goal}} \\
\end{align*}
```

where:
- `J(x, u)` is the objective function to minimize
- `f(·)` represents system dynamics encoded via *integrators*
- `c(·)` represents additional nonlinear constraints
- `x` is the state trajectory
- `u` is the control trajectory

The underlying nonlinear solver is [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl).

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

# Define dynamics
A = [-0.1 1.0; -1.0 -0.1]
B = reshape([0.0, 1.0], 2, 1)
integrator = BilinearIntegrator([A B], traj, :x, :u)

# Define objective
obj = QuadraticRegularizer(:u, traj, 1.0)

# Create and solve problem
prob = DirectTrajOptProblem(traj, obj, integrator)
solve!(prob; max_iter=100)
```

## Key Features

- **Flexible dynamics**: Define system evolution via integrators
- **Modular objectives**: Combine multiple cost terms (regularization, minimum time, etc.)
- **Constraint support**: Bounds, equality, and general nonlinear constraints  
- **Automatic differentiation**: Efficient gradients and Hessians
- **Sparse formulations**: Exploits problem structure for efficiency 


### Building Documentation
This package uses a Documenter config that is shared with many of our other repositories. To build the docs, you will need to run the docs setup script to clone and pull down the utility. 
```
# first time only
./docs/get_docs_utils.sh   # or ./get_docs_utils.sh if cwd is in ./docs/
```

To build the docs pages:
```
julia --project=docs docs/make.jl
```

or editing the docs live:
```
julia --project=docs
> using LiveServer, Piccolo, Revise
> servedocs(literate_dir="docs/literate", skip_dirs=["docs/src/generated", "docs/src/assets/"], skip_files=["docs/src/index.md"])
```

> **Note:** `servedocs` needs to watch a subset of the files in the `docs/` folder. If it watches files that are generated on a docs build/re-build, `servedocs` will continuously try to re-serve the pages.
> 
> To prevent this, ensure all generated files are included in the skip dirs or skip files args for `servedocs`.

For example, if we forget index.md like so:
```
julia --project=docs
> using LiveServer, Piccolo, Revise
> servedocs(literate_dir="docs/literate", skip_dirs=["docs/src/generated", "docs/src/assets/"])
```
it will not build and serve.

-----

*"It seems that perfection is attained not when there is nothing more to add, but when there is nothing more to take away." - Antoine de Saint-Exupéry*