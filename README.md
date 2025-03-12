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
        <a href="https://harmoniqs.github.io/DirectTrajOpt.jl/stable/">
          <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable"/>
        </a>
        <a href="https://harmoniqs.github.io/DirectTrajOpt.jl/dev/">
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
