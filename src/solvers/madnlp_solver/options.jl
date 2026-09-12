export MadNLPOptions

"""
    MadNLPOptions <: Solvers.AbstractSolverOptions

Configuration options for the MadNLP nonlinear solver, as used by the
`MadNLPSolverExt` extension.

Any field can also be passed directly as a keyword argument to `solve!`:
```julia
solve!(prob; options = MadNLPOptions(max_iter = 500))
```

# Commonly used fields
- `tol::Float64 = 1e-8`: Termination tolerance on the KKT residual
- `max_iter::Int = 3000`: Maximum number of solver iterations
- `print_level::Int = 3`: MadNLP output verbosity (MadNLP.LogLevels 1–6)
- `hessian_approximation::String = "exact"`: `"exact"` or `"compact_lbfgs"`
- `linear_solver::Any = nothing`: MadNLP linear-solver type, e.g. `MadNLP.LapackCPUSolver` (`nothing` ⇒ MadNLP default)
- `intermediate_callback::Any = nothing`: `DirectTrajOpt.AbstractIntermediateCallback` or raw `MadNLP.AbstractUserCallback`

# IpoptOptions ↔ MadNLPOptions mapping (the migration table)

Every `IpoptOptions` field is classified — `mapped` (has a counterpart here),
`defaulted` (no field here; the MadNLP-native alternative is named and can be
passed via the raw pass-through of `DirectTrajOpt._solve_with_kwargs`), or
`unsupported` (no MadNLP counterpart; passing it as a kwarg to a MadNLP solve
**warns loudly** — zero silent drops). Machine-checked by the
"exhaustive enumeration" testitem; this table is the source of truth for the
MadNLP-first flip's migration note.

| IpoptOptions field | disposition | MadNLP-native note |
|:-------------------|:------------|:-------------------|
| `tol` | mapped → `tol` | |
| `max_iter` | mapped → `max_iter` | |
| `print_level` | mapped → `print_level` | scale translated (`Int` 0–12 → `MadNLP.LogLevels`) |
| `hessian_approximation` | mapped → `hessian_approximation` | values differ: Ipopt `"limited-memory"` ≡ MadNLP `"compact_lbfgs"` |
| `linear_solver` | mapped → `linear_solver` | type differs: Ipopt `String` (`"mumps"`) ↔ MadNLP `Type` (`MadNLP.MumpsSolver`) |
| `intermediate_callback` | mapped → `intermediate_callback` | same solver-agnostic contract |
| `eval_hessian` | mapped → `hessian_approximation` | kwarg-level: `true` → `"exact"`, `false` → `"compact_lbfgs"` |
| `s_max` | defaulted | MadNLP-native `s_max` |
| `max_cpu_time` | defaulted | MadNLP-native `max_wall_time` (wall, not CPU) |
| `acceptable_tol` | defaulted | MadNLP-native `acceptable_tol` |
| `acceptable_iter` | defaulted | MadNLP-native `acceptable_iter` |
| `diverging_iterates_tol` | defaulted | MadNLP-native `diverging_iterates_tol` |
| `mu_target` | defaulted | MadNLP-native barrier `mu_min` (`MonotoneUpdate`) |
| `nlp_scaling_method` | defaulted | MadNLP-native `nlp_scaling::Bool` + `nlp_scaling_max_gradient` |
| `output_file` | defaulted | MadNLP-native `output_file` |
| `mu_strategy` | defaulted | MadNLP-native adaptive barrier: `barrier = MadNLP.LOQOUpdate(tol, ...)` or `MadNLP.QualityFunctionUpdate(tol, ...)` |
| `adaptive_mu_globalization` | defaulted | rides the same adaptive-barrier types (`barrier`) |
| `refine` | defaulted | MadNLP linear solvers refine internally (e.g. `tol_linear_solve`) |
| `acceptable_obj_change_tol` | defaulted | nearest: `obj_max_inc` (different semantics) |
| `dual_inf_tol` | unsupported | no MadNLP counterpart |
| `constr_viol_tol` | unsupported | no MadNLP counterpart (feasibility folded into `tol`) |
| `compl_inf_tol` | unsupported | no MadNLP counterpart |
| `acceptable_dual_inf_tol` | unsupported | no MadNLP counterpart |
| `acceptable_constr_viol_tol` | unsupported | no MadNLP counterpart |
| `acceptable_compl_inf_tol` | unsupported | no MadNLP counterpart |
| `hsllib` | unsupported | HSL-specific (Ipopt only) |
| `inf_pr_output` | unsupported | display-only; no MadNLP counterpart |
| `print_user_options` | unsupported | no MadNLP counterpart |
| `print_options_documentation` | unsupported | no MadNLP counterpart |
| `print_timing_statistics` | unsupported | no MadNLP counterpart |
| `print_options_mode` | unsupported | no MadNLP counterpart |
| `print_advanced_options` | unsupported | no MadNLP counterpart |
| `print_info_string` | unsupported | no MadNLP counterpart |
| `print_frequency_iter` | unsupported | no MadNLP counterpart |
| `print_frequency_time` | unsupported | no MadNLP counterpart |
| `recalc_y` | unsupported | no MadNLP counterpart |
| `recalc_y_feas_tol` | unsupported | no MadNLP counterpart |
| `watchdog_shortened_iter_trigger` | unsupported | no watchdog in MadNLP 0.9 |
| `watchdog_trial_iter_max` | unsupported | no watchdog in MadNLP 0.9 |

# MadNLP-native-only fields (no IpoptOptions counterpart)
- `array_type::Any = nothing`: e.g. `CUDA.CuArray` for GPU solves
- `kkt_system::Any = nothing`: e.g. `MadNLP.SparseUnreducedKKTSystem`
- `cudss_ordering::Any = nothing`: e.g. `MadNLPGPU.AMD_ORDERING`
- `fixed_variable_treatment::Union{Type,Nothing} = nothing`: `MadNLP.MakeParameter` / `MadNLP.RelaxBound`

# Raw pass-through
`DirectTrajOpt._solve_with_kwargs(prob, options; kwargs...)` forwards kwargs
that match no field here directly into MadNLP's option dict — the escape hatch
for MadNLP-native knobs not on this struct (`barrier`, `bound_push`, …).
Note MadNLP itself silently discards option keys it does not recognize.
The standard `solve!` path instead warns loudly on unmatched kwargs.
"""
Base.@kwdef mutable struct MadNLPOptions <: Solvers.AbstractSolverOptions
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

# ----------------------------------------------------------------------------
# IpoptOptions ↔ MadNLPOptions mapping — machine-checked source of truth
# (C2, spec-20260830-madnlp-first-flip A7/A8). Published in the MadNLPOptions
# docstring; asserted exhaustively by the "exhaustive enumeration" testitem.
# Keep this table and the docstring's table in lockstep.
# ----------------------------------------------------------------------------

"""
Row shape: `(ipopt_field, disposition, target, note)` with
`disposition ∈ (:mapped, :defaulted, :unsupported)`. `mapped` rows carry the
counterpart field on `MadNLPOptions` (or, for the kwarg-level `eval_hessian`
translation handled by the MadNLP extension, its destination field);
`defaulted`/`unsupported` rows name the MadNLP-native alternative (or why
none exists) in `note`.
"""
const IPOPT_TO_MADNLP_OPTIONS = [
    # ── mapped ──
    (ipopt_field = :tol, disposition = :mapped, target = :tol, note = ""),
    (ipopt_field = :max_iter, disposition = :mapped, target = :max_iter, note = ""),
    (
        ipopt_field = :print_level,
        disposition = :mapped,
        target = :print_level,
        note = "scale translated: Ipopt Int 0–12 → MadNLP.LogLevels",
    ),
    (
        ipopt_field = :hessian_approximation,
        disposition = :mapped,
        target = :hessian_approximation,
        note = "values differ: Ipopt \"limited-memory\" ≡ MadNLP \"compact_lbfgs\"",
    ),
    (
        ipopt_field = :linear_solver,
        disposition = :mapped,
        target = :linear_solver,
        note = "type differs: Ipopt String (\"mumps\") ↔ MadNLP Type (MadNLP.MumpsSolver)",
    ),
    (
        ipopt_field = :intermediate_callback,
        disposition = :mapped,
        target = :intermediate_callback,
        note = "same solver-agnostic AbstractIntermediateCallback contract",
    ),
    (
        ipopt_field = :eval_hessian,
        disposition = :mapped,
        target = :hessian_approximation,
        note = "kwarg-level: true → \"exact\", false → \"compact_lbfgs\" (ext _solve)",
    ),
    # ── defaulted (MadNLP-native alternative named) ──
    (
        ipopt_field = :s_max,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP-native s_max",
    ),
    (
        ipopt_field = :max_cpu_time,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP-native max_wall_time (wall, not CPU)",
    ),
    (
        ipopt_field = :acceptable_tol,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP-native acceptable_tol",
    ),
    (
        ipopt_field = :acceptable_iter,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP-native acceptable_iter",
    ),
    (
        ipopt_field = :diverging_iterates_tol,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP-native diverging_iterates_tol",
    ),
    (
        ipopt_field = :mu_target,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP-native barrier mu_min (MonotoneUpdate)",
    ),
    (
        ipopt_field = :nlp_scaling_method,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP-native nlp_scaling::Bool + nlp_scaling_max_gradient",
    ),
    (
        ipopt_field = :output_file,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP-native output_file",
    ),
    (
        ipopt_field = :mu_strategy,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP-native adaptive barrier: barrier = MadNLP.LOQOUpdate(tol, ...) or MadNLP.QualityFunctionUpdate(tol, ...)",
    ),
    (
        ipopt_field = :adaptive_mu_globalization,
        disposition = :defaulted,
        target = nothing,
        note = "rides the same adaptive-barrier types (barrier)",
    ),
    (
        ipopt_field = :refine,
        disposition = :defaulted,
        target = nothing,
        note = "MadNLP linear solvers refine internally (e.g. tol_linear_solve)",
    ),
    (
        ipopt_field = :acceptable_obj_change_tol,
        disposition = :defaulted,
        target = nothing,
        note = "nearest: obj_max_inc (different semantics)",
    ),
    # ── unsupported (loud warn on the kwarg path) ──
    (
        ipopt_field = :dual_inf_tol,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :constr_viol_tol,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart (feasibility folded into tol)",
    ),
    (
        ipopt_field = :compl_inf_tol,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :acceptable_dual_inf_tol,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :acceptable_constr_viol_tol,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :acceptable_compl_inf_tol,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :hsllib,
        disposition = :unsupported,
        target = nothing,
        note = "HSL-specific (Ipopt only)",
    ),
    (
        ipopt_field = :inf_pr_output,
        disposition = :unsupported,
        target = nothing,
        note = "display-only; no MadNLP counterpart",
    ),
    (
        ipopt_field = :print_user_options,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :print_options_documentation,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :print_timing_statistics,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :print_options_mode,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :print_advanced_options,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :print_info_string,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :print_frequency_iter,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :print_frequency_time,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :recalc_y,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :recalc_y_feas_tol,
        disposition = :unsupported,
        target = nothing,
        note = "no MadNLP counterpart",
    ),
    (
        ipopt_field = :watchdog_shortened_iter_trigger,
        disposition = :unsupported,
        target = nothing,
        note = "no watchdog in MadNLP 0.9",
    ),
    (
        ipopt_field = :watchdog_trial_iter_max,
        disposition = :unsupported,
        target = nothing,
        note = "no watchdog in MadNLP 0.9",
    ),
]

# MadNLPOptions fields with no IpoptOptions counterpart (reverse-coverage set
# for the enumeration testitem).
const MADNLP_NATIVE_ONLY_FIELDS =
    (:array_type, :kkt_system, :cudss_ordering, :fixed_variable_treatment)

@testitem "IpoptOptions ↔ MadNLPOptions mapping: exhaustive enumeration, zero silent drops" begin
    using DirectTrajOpt: IpoptSolverExt, MadNLPSolverExtStub

    # The machine-checked source of truth behind the MadNLPOptions docstring's
    # migration table. Every IpoptOptions field is classified exactly once as
    # mapped | defaulted | unsupported; non-mapped rows name the MadNLP-native
    # alternative (or document its absence); every MadNLPOptions field is
    # either a mapped target or explicitly MadNLP-native-only.
    ipopt_fields = fieldnames(IpoptSolverExt.IpoptOptions)
    madnlp_fields = fieldnames(DirectTrajOpt.MadNLPOptions)
    table = MadNLPSolverExtStub.IPOPT_TO_MADNLP_OPTIONS

    dispositions = (:mapped, :defaulted, :unsupported)

    # Exhaustive: every IpoptOptions field classified exactly once.
    @test length(table) == length(ipopt_fields)
    @test Set(r.ipopt_field for r in table) == Set(ipopt_fields)

    targets = Set{Symbol}()
    for r in table
        @test r.disposition in dispositions
        if r.disposition === :mapped
            # Mapped rows resolve to a real MadNLPOptions field — except the
            # kwarg-level eval_hessian → hessian_approximation translation,
            # which the MadNLP extension applies in `_solve` (see the ext's
            # "eval_hessian kwarg routing" testitems).
            @test r.target === :hessian_approximation && r.ipopt_field === :eval_hessian ||
                  r.target in madnlp_fields
            push!(targets, r.target)
        else
            # defaulted / unsupported rows carry a note naming the MadNLP-native
            # alternative (or documenting why none exists).
            @test !isempty(r.note)
        end
    end

    # Reverse coverage: every MadNLPOptions field is either a mapped target or
    # an explicitly-listed MadNLP-native-only field (no Ipopt counterpart).
    @test Set(madnlp_fields) ⊆ targets ∪ Set(MadNLPSolverExtStub.MADNLP_NATIVE_ONLY_FIELDS)
end
