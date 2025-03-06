```@raw html
<div align="center">
  <a href="https://github.com/harmoniqs/Piccolo.jl">
    <img src="assets/logo.svg" alt="Piccolo.jl" width="25%"/>
  </a>
</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>Documentation</b>
        <br>
        <a href="https://harmoniqs.github.io/DirectCollocation.jl/stable/">
          <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable"/>
        </a>
        <a href="https://harmoniqs.github.io/DirectCollocation.jl/dev/">
          <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev"/>
        </a>
      </td>
      <td align="center">
        <b>Build Status</b>
        <br>
        <a href="https://github.com/harmoniqs/DirectCollocation.jl/actions/workflows/CI.yml?query=branch%3Amain">
          <img src="https://github.com/harmoniqs/DirectCollocation.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status"/>
        </a>
        <a href="https://codecov.io/gh/harmoniqs/DirectCollocation.jl">
          <img src="https://codecov.io/gh/harmoniqs/DirectCollocation.jl/branch/main/graph/badge.svg" alt="Coverage"/>
        </a>
      </td>
      <td align="center">
        <b>License</b>
        <br>
        <a href="https://opensource.org/licenses/MIT">
          <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
        </a>
      </td>
      <td align="center">
        <b>Support</b>
        <br>
        <a href="https://unitary.fund">
          <img src="https://img.shields.io/badge/Supported%20By-Unitary%20Fund-FFFF00.svg" alt="Unitary Fund"/>
        </a>
      </td>
    </tr>
  </table>
</div>

<div align="center">
<br>
</div>
```

# DirectCollocation

**DirectCollocation.jl** provides a core library for quantum collocation methods. It is designed to be used in conjunction with the [QuantumCollocation.jl](https://github.com/harmoniqs/QuantumCollocation.jl) package and the [Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl) ecosystem, which provides a high-level interface for solving quantum optimal control problems using direct collocation.

The underlying nonlinear solver is [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), which is a Julia interface to the [Ipopt](https://coin-or.github.io/Ipopt/) solver. 
