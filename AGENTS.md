# AGENTS.md — DirectTrajOpt.jl

DirectTrajOpt.jl is the direct trajectory optimization layer under
Piccolo: Ipopt and MadNLP backends and the solver-agnostic intermediate
callback interface.

## Conventions

- `Pkg.test()` — the test suite carries Aqua, JET, and solver-comparison
  suites (including MadNLP utilities); run the relevant comparison when
  touching solver plumbing.
- Version pins matter to downstream authoring (see lesson 1).

## Toolchain lessons (load-bearing)

1. **The callback interface is version-gated: 0.9.6 vs ≥ 0.9.7.**
   `IpoptOptions(intermediate_callback = …)` **throws at construction**
   on 0.9.6; the solver-agnostic `AbstractIntermediateCallback` and the
   `intermediate_callback` option arrive in 0.9.7. A script authored
   against the wrong pin fails at construction time, not at solve time —
   check the pinned version before writing callback code, and keep
   template code that uses the agnostic interface lockstep with the
   Manifest bump that pins ≥ 0.9.7.

2. **Rich IPM state only rides the raw Ipopt callback.** Per-iteration
   `inf_pr` / `inf_du`-class telemetry needs the Ipopt intermediate
   callback; the solver-agnostic callback cannot carry it. Telemetry
   authors: hang solver-state emission lines on the Ipopt path; use the
   agnostic interface for solver-portable behavior only.

## Provenance

Seeded 2026-08-22 from amicode toolchain context (the bundled template's
per-iter plotting idiom) under the knowledge-routing law: lessons live in
the repo where agents use them. Decisions with history go to ADRs.
