# Changelog

Notable changes to DirectTrajOpt.jl. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes before v0.9.8 are not recorded here — see the
[GitHub releases](https://github.com/harmoniqs/DirectTrajOpt.jl/releases).

## [Unreleased]

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

  On a uniform grid the old *value* is reproduced exactly by passing `R * Δt` —
  though not the old `∂J/∂Δt`, so that is not a drop-in substitution when the
  timestep is a decision variable. Problems with tuned regularisation weights
  should be re-tuned rather than rescaled — the point of the fix is that the old
  quantity was not a quadrature of anything.

  The value, gradient, full Hessian and Hessian sparsity structure were updated
  together. `∂²J/∂Δt²` is now identically zero and is no longer declared as a
  structural nonzero; the variable–timestep cross term is now `R ⊙ (v - v_baseline)`.

  `LinearRegularizer` is unaffected — it already weighted by a single Δt.

### Fixed

- `QuadraticRegularizer`'s `hessian_structure` no longer over-declares the
  control–control block. `R` is a vector of per-component weights, so `∂²J/∂v²` is
  diagonal, but the full `d × d` block was declared — reserving `d(d-1)/2`
  structural nonzeros per knot that can never be nonzero.

- Corrected drifted comments on the `QuadraticRegularizer` Hessian blocks, which
  stated factors the adjacent code did not apply.

### Internal

- `QuadraticRegularizer`'s `gradient!`, `hessian_structure` and `get_full_hessian`
  now guard the `traj.components[traj.timestep]` lookup behind
  `traj.timestep isa Symbol`, as `LinearRegularizer` already did. This is
  defensive only: `NamedTrajectory.timestep` is typed `Symbol` in the current
  NamedTrajectories, so a fixed timestep is not currently representable and the
  branch folds away. The `∂²J/∂v²` block is emitted either way — a fixed Δt would
  still be a factor in the value — so, unlike `LinearRegularizer`, these methods
  cannot simply return early.
