```@raw html
<div align="center">

<a href="https://github.com/harmoniqs/Piccolo.jl">
  <img src="assets/logo.svg" alt="Piccolo.jl" width="25%"/>
</a> 

<div style="display: table; width: 100%;">
  <div style="display: table-row;">
    <div style="display: table-cell; text-align: center;"><b>Documentation</b></div>
    <div style="display: table-cell; text-align: center;"><b>Build Status</b></div>
    <div style="display: table-cell; text-align: center;"><b>Support</b></div>
  </div>
  <div style="display: table-row;">
    <div style="display: table-cell; text-align: center;">
      <a href="https://harmoniqs.github.io/DirectCollocation.jl/stable/">
        <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable"/>
      </a>
    </div>
    <div style="display: table-cell; text-align: center;">
      <a href="https://github.com/harmoniqs/DirectCollocation.jl/actions/workflows/CI.yml?query=branch%3Amain">
        <img src="https://github.com/harmoniqs/DirectCollocation.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status"/>
      </a>
      <a href="https://codecov.io/gh/harmoniqs/DirectCollocation.jl">
        <img src="https://codecov.io/gh/harmoniqs/DirectCollocation.jl/branch/main/graph/badge.svg" alt="Coverage"/>
      </a>
    </div>
    <div style="display: table-cell; text-align: center;">
      <a href="https://unitary.fund">
        <img src="https://img.shields.io/badge/Supported%20By-Unitary%20Fund-FFFF00.svg" alt="Unitary Fund"/>
      </a>
    </div>
  </div>
</div>

<br>

</div>
```

# DirectCollocation

**DirectCollocation.jl** provides a core library for quantum collocation methods. It is designed to be used in conjunction with the [QuantumCollocation.jl](https://github.com/harmoniqs/QuantumCollocation.jl) package and the [Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl) ecosystem, which provides a high-level interface for solving quantum optimal control problems using direct collocation.

The underlying nonlinear solver is [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl), which is a Julia interface to the [Ipopt](https://coin-or.github.io/Ipopt/) solver. 
