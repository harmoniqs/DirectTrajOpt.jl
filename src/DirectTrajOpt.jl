module DirectTrajOpt

using Reexport

# Common interface that both integrators and constraints implement
include("common_interface.jl")
@reexport using .CommonInterface

include("constraints/_constraints.jl")
using .Constraints
# Re-export constraint types but not the interface functions (they come from CommonInterface)
for name in names(Constraints, all=false)
    if name ∉ [:jacobian_structure, :jacobian!, :hessian_structure, :hessian_of_lagrangian, :Constraints]
        @eval export $name
    end
end

include("objectives/_objectives.jl")
@reexport using .Objectives

include("integrators/_integrators.jl")
using .Integrators
# Re-export integrator types but not the interface functions (they come from CommonInterface)
for name in names(Integrators, all=false)
    if name ∉ [:jacobian_structure, :jacobian!, :hessian_structure, :hessian_of_lagrangian, :Integrators]
        @eval export $name
    end
end

include("problems.jl")
@reexport using .Problems

include("solvers.jl")
@reexport using .Solvers

include("solvers/ipopt_solver/IpoptSolverExt.jl")
@reexport using .IpoptSolverExt

end
