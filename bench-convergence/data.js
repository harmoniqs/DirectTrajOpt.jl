window.BENCHMARK_DATA = {
  "lastUpdate": 1780945976982,
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
      },
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
          "id": "204b9ee52ccfcc9bc06575da26654c414bbf947a",
          "message": "ci+docs(benchmarks): run suites on Julia 1.12 + refresh page data (#103)\n\nAll three benchmark suites on Julia 1.12; benchmarks.md tables refreshed with real 1.12 numbers (commit eeba1ff run); dashboard cadence wording corrected to per-release (v* tags). Admin-merge: only red is the pre-existing flaky Hessian CI test.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-08T13:37:25-04:00",
          "tree_id": "c2231401fbacdb4cbc1d5d7109466e2febb52f29",
          "url": "https://github.com/harmoniqs/DirectTrajOpt.jl/commit/204b9ee52ccfcc9bc06575da26654c414bbf947a"
        },
        "date": 1780940350265,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "xgate_convergence_ipopt_N51 [wall]",
            "value": 19.108915239,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [alloc]",
            "value": 4311642552,
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
            "value": 18.621993404,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [alloc]",
            "value": 4476556112,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [infidelity]",
            "value": 3.086420008457935e-14,
            "unit": "infidelity"
          }
        ]
      },
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
          "id": "764d3b3b46ba49387e16f38c53b4448f6f220316",
          "message": "ci(alloc): bump alloc-profile timeout 90 -> 120 min for Julia 1.12 (#104)\n\nThe allocation profile runs each solve under Profile.Allocs sampling (~30-40 min\nper solve). On Julia 1.11 the full run was ~76 min; on 1.12 it overran the 90 min\ncap (cancelled at 91 min), so /bench-alloc never seeded. Bump to 120 min to give\nthe slower 1.12 runtime real headroom.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-08T15:10:55-04:00",
          "tree_id": "310483f7b9e5751e9703a98715f52b722ff8e01f",
          "url": "https://github.com/harmoniqs/DirectTrajOpt.jl/commit/764d3b3b46ba49387e16f38c53b4448f6f220316"
        },
        "date": 1780945975113,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "xgate_convergence_ipopt_N51 [wall]",
            "value": 18.784076357,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [alloc]",
            "value": 4311559576,
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
            "value": 18.271785741,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [alloc]",
            "value": 4476613400,
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