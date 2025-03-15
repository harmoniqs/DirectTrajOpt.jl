var documenterSearchIndex = {"docs":
[{"location":"generated/explanation/","page":"Explanation","title":"Explanation","text":"EditURL = \"../../literate/explanation.jl\"","category":"page"},{"location":"generated/explanation/#Quickstart-Guide","page":"Explanation","title":"Quickstart Guide","text":"","category":"section"},{"location":"generated/explanation/#Installation","page":"Explanation","title":"Installation","text":"","category":"section"},{"location":"generated/explanation/","page":"Explanation","title":"Explanation","text":"using DirectTrajOpt","category":"page"},{"location":"generated/explanation/#This-package-also-provides-various-objects-and-bindings-used-in-Quantum-Optimal-Control-problems.","page":"Explanation","title":"This package also provides various objects and bindings used in Quantum Optimal Control problems.","text":"","category":"section"},{"location":"generated/explanation/","page":"Explanation","title":"Explanation","text":"This means various common constraints, integrators, objectives, and losses. This package also provides interfaces for the commonly needed dynamics, and evaluator objects to provide to the solver - which for now is Ipopt.","category":"page"},{"location":"generated/explanation/","page":"Explanation","title":"Explanation","text":"","category":"page"},{"location":"generated/explanation/","page":"Explanation","title":"Explanation","text":"This page was generated using Literate.jl.","category":"page"},{"location":"lib/#Library","page":"Lib","title":"Library","text":"","category":"section"},{"location":"lib/","page":"Lib","title":"Lib","text":"CollapsedDocStrings = true","category":"page"},{"location":"lib/#Constraints","page":"Lib","title":"Constraints","text":"","category":"section"},{"location":"lib/","page":"Lib","title":"Lib","text":"Modules = [DirectTrajOpt.Constraints]","category":"page"},{"location":"lib/#DirectTrajOpt.Constraints.EqualityConstraint","page":"Lib","title":"DirectTrajOpt.Constraints.EqualityConstraint","text":"struct EqualityConstraint\n\nRepresents a linear equality constraint.\n\nFields\n\nts::AbstractArray{Int}: the time steps at which the constraint is applied\njs::AbstractArray{Int}: the components of the trajectory at which the constraint is applied\nvals::Vector{R}: the values of the constraint\nvardim::Int: the dimension of a single time step of the trajectory\nlabel::String: a label for the constraint\n\n\n\n\n\n","category":"type"},{"location":"lib/#DirectTrajOpt.Constraints.EqualityConstraint-Tuple{Symbol, AbstractVector{Int64}, Vector{Float64}, NamedTrajectories.StructNamedTrajectory.NamedTrajectory}","page":"Lib","title":"DirectTrajOpt.Constraints.EqualityConstraint","text":"EqualityConstraint(\n    name::Symbol,\n    ts::Vector{Int},\n    val::Vector{Float64},\n    traj::NamedTrajectory;\n    label=\"equality constraint on trajectory variable [name]\"\n)\n\nConstructs equality constraint for trajectory variable in NamedTrajectory\n\n\n\n\n\n","category":"method"},{"location":"lib/#DirectTrajOpt.Constraints.GlobalEqualityConstraint-Tuple{Symbol, Vector{Float64}, NamedTrajectories.StructNamedTrajectory.NamedTrajectory}","page":"Lib","title":"DirectTrajOpt.Constraints.GlobalEqualityConstraint","text":"GlobalEqualityConstraint(\n    name::Symbol,\n    val::Vector{Float64},\n    traj::NamedTrajectory;\n    label=\"equality constraint on global variable [name]\"\n)::EqualityConstraint\n\nConstructs equality constraint for global variable in NamedTrajectory\n\n\n\n\n\n","category":"method"},{"location":"lib/#Integrators","page":"Lib","title":"Integrators","text":"","category":"section"},{"location":"lib/","page":"Lib","title":"Lib","text":"Modules = [DirectTrajOpt.Integrators]","category":"page"},{"location":"lib/#DirectTrajOpt.Integrators.dense-Tuple{Any, Any, Any}","page":"Lib","title":"DirectTrajOpt.Integrators.dense","text":"dense(vals, structure, shape)\n\nConvert sparse data to dense matrix.\n\nArguments\n\nvals: vector of values\nstructure: vector of tuples of indices\nshape: tuple of matrix dimensions\n\n\n\n\n\n","category":"method"},{"location":"lib/#DirectTrajOpt.Integrators.show_diffs-Tuple{AbstractMatrix, AbstractMatrix}","page":"Lib","title":"DirectTrajOpt.Integrators.show_diffs","text":"show_diffs(A::Matrix, B::Matrix)\n\nShow differences between matrices.\n\n\n\n\n\n","category":"method"},{"location":"lib/#Objectives","page":"Lib","title":"Objectives","text":"","category":"section"},{"location":"lib/","page":"Lib","title":"Lib","text":"Modules = [DirectTrajOpt.Objectives]","category":"page"},{"location":"lib/#DirectTrajOpt.Objectives.Objective","page":"Lib","title":"DirectTrajOpt.Objectives.Objective","text":"Objective\n\nA structure for defining objective functions.\n\nFields:     L: the objective function     ∇L: the gradient of the objective function     ∂²L: the Hessian of the objective function     ∂²L_structure: the structure of the Hessian of the objective function\n\n\n\n\n\n","category":"type"},{"location":"lib/#DirectTrajOpt.Objectives.MinimumTimeObjective-Tuple{NamedTrajectories.StructNamedTrajectory.NamedTrajectory}","page":"Lib","title":"DirectTrajOpt.Objectives.MinimumTimeObjective","text":"MinimumTimeObjective\n\nA type of objective that counts the time taken to complete a task.  D is a scaling factor.\n\n\n\n\n\n","category":"method"},{"location":"lib/#DirectTrajOpt.Objectives.QuadraticRegularizer-Tuple{Symbol, NamedTrajectories.StructNamedTrajectory.NamedTrajectory, AbstractVector{<:Real}}","page":"Lib","title":"DirectTrajOpt.Objectives.QuadraticRegularizer","text":"QuadraticRegularizer\n\nA quadratic regularizer for a trajectory component.\n\nFields:     name: the name of the trajectory component to regularize     traj: the trajectory     R: the regularization matrix diagonal     baseline: the baseline values for the trajectory component     times: the times at which to evaluate the regularizer\n\n\n\n\n\n","category":"method"},{"location":"lib/#Dynamics","page":"Lib","title":"Dynamics","text":"","category":"section"},{"location":"lib/","page":"Lib","title":"Lib","text":"Modules = [DirectTrajOpt.Dynamics]","category":"page"},{"location":"lib/#DirectTrajOpt.Dynamics.TrajectoryDynamics","page":"Lib","title":"DirectTrajOpt.Dynamics.TrajectoryDynamics","text":"TrajectoryDynamics\n\nA struct for trajectory optimization dynamics, represented by integrators that compute single time step dynamics, and functions for jacobians and hessians.\n\nFields\n\nintegrators::Union{Nothing, Vector{<:AbstractIntegrator}}: Vector of integrators.\nF!::Function: Function to compute trajectory dynamics.\n∂F!::Function: Function to compute the Jacobian of the dynamics.\n∂fs::Vector{SparseMatrixCSC{Float64, Int}}: Vector of Jacobian matrices.\nμ∂²F!::Union{Function, Nothing}: Function to compute the Hessian of the Lagrangian.\nμ∂²fs::Vector{SparseMatrixCSC{Float64, Int}}: Vector of Hessian matrices.\ndim::Int: Total dimension of the dynamics.\n\n\n\n\n\n","category":"type"},{"location":"lib/#Problems","page":"Lib","title":"Problems","text":"","category":"section"},{"location":"lib/","page":"Lib","title":"Lib","text":"Modules = [DirectTrajOpt.Problems]","category":"page"},{"location":"lib/#DirectTrajOpt.Problems.DirectTrajOptProblem","page":"Lib","title":"DirectTrajOpt.Problems.DirectTrajOptProblem","text":"mutable struct DirectTrajOptProblem <: AbstractProblem\n\nStores all the information needed to set up and solve a DirectTrajOptProblem as well as the solution after the solver terminates.\n\nFields\n\noptimizer::Ipopt.Optimizer: Ipopt optimizer object\n\n\n\n\n\n","category":"type"},{"location":"lib/#DirectTrajOpt.Problems.get_trajectory_constraints-Tuple{NamedTrajectories.StructNamedTrajectory.NamedTrajectory}","page":"Lib","title":"DirectTrajOpt.Problems.get_trajectory_constraints","text":"trajectory_constraints(traj::NamedTrajectory)\n\nImplements the initial and final value constraints and bounds constraints on the controls and states as specified by traj.\n\n\n\n\n\n","category":"method"},{"location":"lib/#Problem-Solvers","page":"Lib","title":"Problem Solvers","text":"","category":"section"},{"location":"lib/","page":"Lib","title":"Lib","text":"Modules = [DirectTrajOpt.Solvers]","category":"page"},{"location":"lib/#Problem-Solvers-2","page":"Lib","title":"Problem Solvers","text":"","category":"section"},{"location":"lib/","page":"Lib","title":"Lib","text":"Modules = [DirectTrajOpt.IpoptSolverExt]","category":"page"},{"location":"lib/#DirectTrajOpt.IpoptSolverExt.IpoptOptions","page":"Lib","title":"DirectTrajOpt.IpoptSolverExt.IpoptOptions","text":"Solver options for Ipopt\n\nhttps://coin-or.github.io/Ipopt/OPTIONS.html#OPT_print_options_documentation\n\n\n\n\n\n","category":"type"},{"location":"lib/#DirectTrajOpt.Solvers.solve!-Tuple{DirectTrajOptProblem}","page":"Lib","title":"DirectTrajOpt.Solvers.solve!","text":"solve!(prob::DirectTrajOptProblem;         inittraj=nothing,         savepath=nothing,         maxiter=prob.ipoptoptions.maxiter,         linearsolver=prob.ipoptoptions.linearsolver,         printlevel=prob.ipoptoptions.printlevel,         removeslackvariables=false,         callback=nothing         # statetype=:unitary,         # print_fidelity=false,     )\n\nCall optimization solver to solve the quantum control problem with parameters and callbacks.\n\nArguments\n\nprob::DirectTrajOptProblem: The quantum control problem to solve.\ninit_traj::NamedTrajectory: Initial guess for the control trajectory. If not provided, a random guess will be generated.\nsave_path::String: Path to save the problem after optimization.\nmax_iter::Int: Maximum number of iterations for the optimization solver.\nlinear_solver::String: Linear solver to use for the optimization solver (e.g., \"mumps\", \"paradiso\", etc).\nprint_level::Int: Verbosity level for the solver.\ncallback::Function: Callback function to call during optimization steps.\n\n\n\n\n\n","category":"method"},{"location":"#DirectTrajOpt.jl","page":"Home","title":"DirectTrajOpt.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"<!-- <div align=\"center\">\n  <a href=\"https://github.com/harmoniqs/Piccolo.jl\">\n    <img src=\"assets/logo.svg\" alt=\"Piccolo.jl\" width=\"25%\"/>\n  </a>\n</div> -->\n\n<div align=\"center\">\n  <table>\n    <tr>\n      <td align=\"center\">\n        <b>Documentation</b>\n        <br>\n        <a href=\"https://harmoniqs.github.io/DirectTrajOpt.jl/stable/\">\n          <img src=\"https://img.shields.io/badge/docs-stable-blue.svg\" alt=\"Stable\"/>\n        </a>\n        <a href=\"https://harmoniqs.github.io/DirectTrajOpt.jl/dev/\">\n          <img src=\"https://img.shields.io/badge/docs-dev-blue.svg\" alt=\"Dev\"/>\n        </a>\n      </td>\n      <td align=\"center\">\n        <b>Build Status</b>\n        <br>\n        <a href=\"https://github.com/harmoniqs/DirectTrajOpt.jl/actions/workflows/CI.yml?query=branch%3Amain\">\n          <img src=\"https://github.com/harmoniqs/DirectTrajOpt.jl/actions/workflows/CI.yml/badge.svg?branch=main\" alt=\"Build Status\"/>\n        </a>\n        <a href=\"https://codecov.io/gh/harmoniqs/DirectTrajOpt.jl\">\n          <img src=\"https://codecov.io/gh/harmoniqs/DirectTrajOpt.jl/branch/main/graph/badge.svg\" alt=\"Coverage\"/>\n        </a>\n      </td>\n      <td align=\"center\">\n        <b>License</b>\n        <br>\n        <a href=\"https://opensource.org/licenses/MIT\">\n          <img src=\"https://img.shields.io/badge/License-MIT-yellow.svg\" alt=\"MIT License\"/>\n        </a>\n    </tr>\n  </table>\n</div>\n\n<div align=\"center\">\n<br>\n</div>","category":"page"},{"location":"","page":"Home","title":"Home","text":"DirectTrajOpt.jl provides abstractions and utilities for setting up and solving direct trajectory optimization problems of the form:","category":"page"},{"location":"","page":"Home","title":"Home","text":"$","category":"page"},{"location":"","page":"Home","title":"Home","text":"\\begin{align} \\underset{x{1:N}, u{1:N-1}}{\\text{minimize}} \\quad & J(x{1:N}, u{1:N-1}) \\\n\\text{subject to} \\quad & f(x{k+1}, xk, uk, \\Delta t, tk) = 0\\\n& ck(xk, uk) \\geq 0 \\\n& x1 = x_{\\text{init}} \\\n\\end{align} $","category":"page"},{"location":"","page":"Home","title":"Home","text":"where J(x_1N u_1N-1) is a user-defined cost function, f(x_k+1 x_k u_k Delta t t_k) is an integrator funtion encoding the dynamics of the system, and c_k(x_k u_k) are user-defined constraints.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The underlying nonlinear solver is Ipopt.jl, which is a Julia interface to the Ipopt solver. ","category":"page"}]
}
