window.BENCHMARK_DATA = {
  "lastUpdate": 1780936818504,
  "repoUrl": "https://github.com/harmoniqs/DirectTrajOpt.jl",
  "entries": {
    "DirectTrajOpt.jl convergence": [
      {
        "commit": {
          "author": {
            "email": "43344745+jack-champagne@users.noreply.github.com",
            "name": "Jack Champagne",
            "username": "jack-champagne"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7adc3d5ead96581b2f2c5c08a3d009999364eabc",
          "message": "ci(benchmarks): publish dashboards on v* tags, not refs/heads/main (#102)\n\nThe three benchmark workflows trigger only on `push: tags:['v*']` + pull_request\n+ workflow_dispatch (never on push to main), but save-data-file / auto-push were\ngated to `github.ref == 'refs/heads/main'`. Those conditions are mutually\nexclusive, so the gh-pages series was NEVER published — /bench, /bench-alloc and\n/bench-convergence stayed empty (zero github-action-benchmark commits on\ngh-pages, confirmed).\n\nGate on tag refs instead (`startsWith(github.ref, 'refs/tags/v')`), matching the\nactual trigger: each release tag appends one data point; PR runs still render a\ncomparison comment without polluting the series. Per-release rather than\nper-commit, by design (avoids running the heavy suites on every main merge).\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-08T12:08:14-04:00",
          "tree_id": "72dc3aa79e38e5af1e06b870dec2dcab6077db9e",
          "url": "https://github.com/harmoniqs/DirectTrajOpt.jl/commit/7adc3d5ead96581b2f2c5c08a3d009999364eabc"
        },
        "date": 1780936817097,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "xgate_convergence_ipopt_N51 [wall]",
            "value": 18.652418176,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [alloc]",
            "value": 4312604032,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [iters]",
            "value": 22,
            "unit": "iterations"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [infidelity]",
            "value": 4.429490108037726e-11,
            "unit": "infidelity"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [wall]",
            "value": 18.493782763,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [alloc]",
            "value": 4476617640,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [infidelity]",
            "value": 3.086420008457935e-14,
            "unit": "infidelity"
          }
        ]
      }
    ]
  }
}