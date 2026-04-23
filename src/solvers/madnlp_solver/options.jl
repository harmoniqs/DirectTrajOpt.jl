export MadNLPOptions

@kwdef mutable struct MadNLPOptions <: Solvers.AbstractSolverOptions
    # Primary options
    tol::Float64 = 1e-8
    max_iter::Int = 3000
    print_level::Int = 3 # (MadNLP.TRACE::MadNLP.LogLevels = 1, ..., MadNLP.ERROR::MadNLP.LogLevels = 6)
    hessian_approximation::String = "exact" # (exact = MadNLP.ExactHessian, compact_lbfgs = MadNLP.CompactLBFGS) # no other QN methods supported in conjunction with MadNLP.SparseCallback

    # Pass-throughs consumed by MadNLP's MOI layer (not by MadNLP itself);
    # leave as `nothing` to use MadNLP defaults. Only forwarded when non-nothing.
    linear_solver::Any  = nothing  # e.g. MadNLPGPU.CUDSSSolver, MadNLP.LapackCPUSolver
    array_type::Any     = nothing  # e.g. CUDA.CuArray for GPU
    kkt_system::Any     = nothing  # e.g. MadNLP.SparseUnreducedKKTSystem
    cudss_ordering::Any = nothing  # e.g. MadNLPGPU.AMD_ORDERING

    # # Only supported by DirectTrajOpt._solve, as an optional kwarg override of `hessian_approximation`;
    # #   `hessian_approximation = eval_hessian ? "exact" : "compact_lbfgs"`
    # eval_hessian::Bool = true
end
