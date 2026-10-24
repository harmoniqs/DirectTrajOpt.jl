# Changelog

Notable changes to DirectTrajOpt.jl. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes before v0.9.8 are not recorded here — see the
[GitHub releases](https://github.com/harmoniqs/DirectTrajOpt.jl/releases).

## [Unreleased]

## [0.11.0] — 2026-10-02

### Changed

- **MadNLP is the default solver backend (#155)** — the no-kwarg `solve!(prob)` now dispatches MadNLP. MadNLP is a HARD dependency (the `MadNLPSolverExt` weakdep package extension moved from `ext/` into `src/` beside `IpoptSolverExt`); Ipopt remains a hard dependency and fully selectable via `solve!(prob; options = IpoptOptions(...))` or `Solvers._set_DefaultSolverOptions(IpoptSolverExt.IpoptOptions)`. `test/compare_solvers.jl` is a live harness again: a default-leg vs Ipopt-leg comparison (plus an explicit MadNLP leg) with fixed seeds.

- **`MadNLPOptions.barrier` defaults to the restoration-audit verdict: `QualityFunctionUpdate` (#155)** — MadNLP 0.9's native `MonotoneUpdate` barrier stalls on the standard Piccolo QCP class on ~1-in-3 seeds (engages feasibility restoration + robust mode, the barrier re-anchors at mu ≈ 1.5e-4, and at a 60-iter budget the rollout fidelity misses the F ≥ 0.999 bar: seed2 X-gate F = 0.99835 vs Ipopt's 0.99999998; Ipopt never restores on the same cells). Deterministic and machine-independent (identical traces on x86/loaded and arm64/quiet; identical under the stock-Bilinear and HermitianExponential integrators on ZOH pulses). The audit swept 13 option knobs on the seed2 instance (barrier strategies, dual initialization, restoration thresholds, SOC, KKT system, regularization, scaling, bound push, step fraction): the adaptive `MadNLP.QualityFunctionUpdate` barrier avoids the stall at equal wall cost — seed2 F = 0.99999996 at 60 iters (0.99999999 at 100), ZERO restoration engagement, parity-or-better on every audited clean cell (seeds 1/3 stock + HE + free-phase lanes, all ≥ 0.99999987; free-phase lane 0.99999999 across seeds). `barrier = nothing` (the new field's default) resolves at solve time to `MadNLP.QualityFunctionUpdate(tol, 10.0)` — the same `mu_min = min(1e-4, tol)/(barrier_tol_factor+1)` coupling MadNLP applies natively, so a tighter `tol` still lowers the barrier floor; an explicit `MadNLP.AbstractBarrierUpdate` (e.g. `MadNLP.MonotoneUpdate(tol, 10.0)`) passes through untouched. Residual risk, recorded honestly: the audit covered the Piccolo QCP class (stock ZOH + HE + free-phase, seeds 1–3) plus the full DTO suite under the new default; min-time-compression chains, ensemble robustness, and composite-unitary families were not audited cell-by-cell. Audit artifacts: `/tmp/opencode/p1-audit-results*/` (sweep + winners + regression-guard JSONs).

### Fixed

- **The Jacobian-product seam (`evaluator.jl`)** — `MOI.eval_constraint_jacobian_product` / `eval_constraint_jacobian_transpose_product` compute J·w / Jᵀw from the evaluator's cached structure and a reusable pre-allocated values buffer instead of assembling a fresh `zeros(nnz)` (and a `@warn`) per call. Exact math unchanged; MadNLP's restoration/robust path is the beneficiary (its `jtprod!` routes through the transpose product). Verified: 660,960 → 628,360 bytes/call on the standard test problem (the fill-path floor is 632,264), zero warnings on clean solves.

### Added

- **AMICODE_ITER telemetry contract on the MadNLP arm, encoded in the suite** — a raw `MadNLP.AbstractUserCallback` verifier (all four state columns — `cnt.k`, `obj_val`, `inf_pr`, `inf_du` — finite on every `UserCallbackRegular` emission, iters monotone) and a unit test pinning `_MadNLPCallbackAdapter`'s mode filter (restore/robust phases never forward to the solver-agnostic callback). The adapter's `solver` argument is now duck-typed (`variable(x)` + `cnt.k`) — MadNLP invokes callbacks untyped, so behavior is unchanged; it only permits deterministic testing of the filter.

## [0.10.1] — 2026-08-21

### Added

- **Multi-state `BilinearIntegrator` constructor restored** — `BilinearIntegrator(G, xs::AbstractVector{Symbol}, u, traj)` returns (fixes #138): the `c9fdeb7` integrator refactor had dropped the historical `xs` form, leaving every stacked-state caller — concretely Piccolo's exported `VariationalKetIntegrator`/`VariationalUnitaryIntegrator` (Piccolo #300) — with a `MethodError` on construction. The struct gains `x_names::Vector{Symbol}` (following the convention `get_nonlinear_constraints` and Piccolissimo's exponential family already use); `x_name::Symbol` remains as the primary (first) name so existing field access keeps working, and the single-state constructor delegates via `[x]`. `evaluate!`/`eval_jacobian`/`eval_hessian_of_lagrangian` gather the stacked state across all names; `get_nonlinear_constraints` checks `x_names` before `x_name` so an integrator carrying both fields sums the whole stack. Verified bit-equivalent against the single-component reference on coinciding flat data (residuals, Jacobians, Lagrangian Hessians).

### Changed

- Benchmark and convergence environments pin `HarmoniqsBenchmarks` at `v0.2.1` (was a stale rev whose `DirectTrajOpt = "0.9"` compat made both suites unsatisfiable once 0.10.0 reached General).

## [0.10.0] — 2026-08-20

### Added

- **`solve!` returns `SolveStats`** — termination status (MOI code), raw status
  string, NLP objective, IPM iterations, solve wall time, and solver symbol —
  from both the Ipopt and MadNLP paths. Previously both paths ended in
  `return nothing` after `MOI.optimize!`, discarding everything the solver
  knew; callers re-parsed stdout or installed callbacks to learn whether a
  solve converged. Additive in practice (code that ignored the return still
  works). ([#133](https://github.com/harmoniqs/DirectTrajOpt.jl/pull/133))

### Changed (behaviour — see the entry below for the headline)

- Notation pass across docs and docstrings: "knot points" for N, "timestep"
  reserved for the per-knot Δt. ([#131](https://github.com/harmoniqs/DirectTrajOpt.jl/pull/131))

### Housekeeping

- Coverage campaign: 86.45% → **99.45%** line coverage, 59 new tests; 2
  unreachable debug branches removed; the historical vector-syntax
  finite-difference test de-flaked at the root (its `norm(a) − 1.0` fixture
  was kinky at zero — finite-difference Hessians across a kink are unstable;
  replaced by a smooth fixture with a tighter tolerance).
  ([#135](https://github.com/harmoniqs/DirectTrajOpt.jl/pull/135),
  [#136](https://github.com/harmoniqs/DirectTrajOpt.jl/pull/136))


### Changed

- **Behaviour change — `QuadraticRegularizer` now weights each knot by a single
  Δt instead of Δt².**
  ([#122](https://github.com/harmoniqs/DirectTrajOpt.jl/issues/122))

  **Existing `R` values change meaning.** The previous Δt² weighting was not the
  form documented in the docstring, and it made the penalty fall off as `1/N`
  under grid refinement — measured 32× weaker across a 25 → 800 knot sweep for a
  fixed continuous pulse. Degrees of freedom grew linearly while the penalty
  against using them weakened, silently. Any hand-tuned `R` was therefore tuned
  against a grid-dependent quantity.

  On a uniform grid the old value is reproduced exactly by passing `R * Δt`, but
  problems with tuned regularisation weights should be re-tuned rather than
  rescaled — the point of the fix is that the old quantity was not a quadrature
  of anything.

  The value, gradient, full Hessian and Hessian sparsity structure were updated
  together. `∂²J/∂Δt²` is now identically zero and is no longer declared as a
  structural nonzero; the variable–timestep cross term is now `R ⊙ (v - v_baseline)`.

  `LinearRegularizer` is unaffected — it already weighted by a single Δt.

### Fixed

- Corrected drifted comments on the `QuadraticRegularizer` Hessian blocks, which
  stated factors the adjacent code did not apply.
