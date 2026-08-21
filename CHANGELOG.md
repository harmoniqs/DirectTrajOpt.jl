# Changelog

Notable changes to DirectTrajOpt.jl. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes before v0.9.8 are not recorded here — see the
[GitHub releases](https://github.com/harmoniqs/DirectTrajOpt.jl/releases).

## [Unreleased]

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
