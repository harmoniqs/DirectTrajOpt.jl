export MadNLPOptions

@kwdef mutable struct MadNLPOptions <: Solvers.AbstractSolverOptions
    # Primary options
    tol::Float64 = 1e-8
    max_iter::Int = 3000
    print_level::Int = 3 # (MadNLP.TRACE::MadNLP.LogLevels = 1, ..., MadNLP.ERROR::MadNLP.LogLevels = 6)
    hessian_approximation::String = "exact" # (exact = MadNLP.ExactHessian, compact_lbfgs = MadNLP.CompactLBFGS) # no other QN methods supported in conjunction with MadNLP.SparseCallback

    # Pass-throughs consumed by MadNLP's MOI layer (not by MadNLP itself);
    # leave as `nothing` to use MadNLP defaults. Only forwarded when non-nothing.
    linear_solver::Any = nothing  # e.g. MadNLPGPU.CUDSSSolver, MadNLP.LapackCPUSolver
    array_type::Any = nothing  # e.g. CUDA.CuArray for GPU
    kkt_system::Any = nothing  # e.g. MadNLP.SparseUnreducedKKTSystem
    cudss_ordering::Any = nothing  # e.g. MadNLPGPU.AMD_ORDERING

    # Per-iteration user callback. Two accepted forms:
    #
    #   1. A subtype of `DirectTrajOpt.AbstractIntermediateCallback` (solver-agnostic).
    #      Signature: `(cb)(primal::AbstractVector, iter::Integer) -> Bool`.
    #      The MadNLP extension wraps it in an internal adapter at solve time.
    #
    #   2. A raw `MadNLP.AbstractUserCallback` subtype with native MadNLP signature
    #      `(cb)(solver::MadNLP.AbstractMadNLPSolver, mode) -> Bool` — passed through
    #      unwrapped for users who want full access to the IPM state.
    #
    # Return `false` to stop the solver (yields `USER_REQUESTED_STOP`).
    intermediate_callback::Any = nothing

    # Controls how MadNLP handles variables with `lb == ub`. Mirrors MadNLP's
    # own `fixed_variable_treatment::Type` field — must be a `Type` (typically
    # `MadNLP.MakeParameter` or `MadNLP.RelaxBound`). Default (`nothing`) defers
    # to MadNLP's kkt_system-aware conditional default:
    #
    #     kkt_system <: SparseCondensedKKTSystem ? RelaxBound : MakeParameter
    #
    # When an `AbstractIntermediateCallback` is installed and this field is
    # left at `nothing`, `set_options!` only overrides to `RelaxBound` if
    # MadNLP's conditional default would otherwise be `MakeParameter` (which
    # eliminates fixed boundary vars from `solver.x` and breaks trajectory
    # reconstruction). The conditional default's `RelaxBound` branch is left
    # untouched.
    fixed_variable_treatment::Union{Type,Nothing} = nothing

    # # Only supported by DirectTrajOpt._solve, as an optional kwarg override of `hessian_approximation`;
    # #   `hessian_approximation = eval_hessian ? "exact" : "compact_lbfgs"`
    # eval_hessian::Bool = true
end
