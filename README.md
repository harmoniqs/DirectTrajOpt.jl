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

**DirectTrajOpt.jl** provides abstractions and utilities for setting up and solving direct trajectory optimization problems of the form:

$$
\begin{align*}
\underset{x_{1:N}, u_{1:N-1}}{\text{minimize}} \quad & J(x_{1:N}, u_{1:N-1}) \\
\text{subject to} \quad & f(x_{k+1}, x_k, u_k, \Delta t, t_k) = 0\\
& c_k(x_k, u_k) \geq 0 \\
& x_1 = x_{\text{init}} \\
\end{align*}
$$

where $J(x_{1:N}, u_{1:N-1})$ is a user-defined cost function, $f(x_{k+1}, x_k, u_k, \Delta t, t_k)$ is an *integrator* funtion encoding the dynamics of the system, and $c_k(x_k, u_k)$ are user-defined constraints.

The underlying nonlinear solver is [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), which is a Julia interface to the [Ipopt](https://coin-or.github.io/Ipopt/) solver. 


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

## NOTE:
servedocs needs to watch a subset of the files in the `docs/` folder. If it watches files that are generated on a docs build/re-build, servedocs will continuously try to reserve the pages.

To prevent this, ensure all generated files are included in the skip dirs or skip files args for servedocs.

For example, if we forget index.md like so:
```
julia --project=docs
> using LiveServer, Piccolo, Revise
> servedocs(literate_dir="docs/literate", skip_dirs=["docs/src/generated", "docs/src/assets/"])
```
it will not build and serve.

-----

*"It seems that perfection is attained not when there is nothing more to add, but when there is nothing more to take away." - Antoine de Saint-Exupéry*