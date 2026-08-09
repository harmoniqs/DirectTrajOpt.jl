# Contributing to DirectTrajOpt.jl

Welcome! DirectTrajOpt.jl provides the low-level optimization infrastructure
for [Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl) — trajectory
parametrizations, integrators, and solver backends. Licensed under MIT.

Every code change needs a GitHub issue and a PR. Create one first if it does
not exist.

## Dev Setup

```bash
git clone git@github.com:harmoniqs/DirectTrajOpt.jl.git && cd DirectTrajOpt.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Running Tests

```bash
julia --project test/runtests.jl
```

## Submitting a PR

1. Create an issue describing the change.
2. Branch from `main`.
3. Commit focused changes with descriptive messages.
4. Run the test suite.
5. Open a draft PR with `Closes #<N>` at the first commit.
6. Mark ready when green.

## Getting Help

Use [GitHub Discussions](https://github.com/harmoniqs/DirectTrajOpt.jl/discussions) for questions.
