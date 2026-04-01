export MadNLPOptions

@kwdef mutable struct MadNLPOptions <: Solvers.AbstractSolverOptions
    # Primary options
    tol::Float64 = 1e-8
    max_iter::Int = 1000
    print_level::Int = 3 # corresponds to MadNLP.LogLevels(print_level) where TRACE = 1 <= print_level <= 6 = ERROR
    eval_hessian::Bool = true
end
