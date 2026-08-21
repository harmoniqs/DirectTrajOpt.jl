window.BENCHMARK_DATA = {
  "lastUpdate": 1787333530628,
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
          "id": "0c47f87863cb8cf0c685222347f49a5431089cf2",
          "message": "chore: autoformat convergence.jl (MadNLPOptions one-liner) (#110)\n\nJuliaFormatter collapses the MadNLPOptions(...) call onto a single line; my #109\nmulti-line form tripped the Formatter check. No code change.\n\nCo-authored-by: Claude Opus 4.8 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-06-10T00:44:08-04:00",
          "tree_id": "ccd269fd67bf08b4b4ae789e3c39c44de254c596",
          "url": "https://github.com/harmoniqs/DirectTrajOpt.jl/commit/0c47f87863cb8cf0c685222347f49a5431089cf2"
        },
        "date": 1781068737035,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "xgate_convergence_ipopt_N51 [wall]",
            "value": 18.265534953,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [alloc]",
            "value": 4303314520,
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
            "value": 18.005350134,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [alloc]",
            "value": 4477810360,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [iters]",
            "value": 30,
            "unit": "iterations"
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
          "id": "134c3ac47f0f8e862358ce52f4f0570dc3112ec7",
          "message": "Merge pull request #118 from harmoniqs/chore/version-0.9.7\n\nchore: bump version to 0.9.7",
          "timestamp": "2026-07-01T05:14:15-04:00",
          "tree_id": "c722d301a8d064de5d9ab67e5bc8a1dff893683a",
          "url": "https://github.com/harmoniqs/DirectTrajOpt.jl/commit/134c3ac47f0f8e862358ce52f4f0570dc3112ec7"
        },
        "date": 1782899307586,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "xgate_convergence_ipopt_N51 [wall]",
            "value": 18.629796041,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [alloc]",
            "value": 4306633264,
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
            "value": 18.492935329,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [alloc]",
            "value": 4468126288,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [iters]",
            "value": 30,
            "unit": "iterations"
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
            "email": "41800392+gennadiryan@users.noreply.github.com",
            "name": "Gennadi Ryan",
            "username": "gennadiryan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0cd7923e79376e6bf26c3532246169addbc4a424",
          "message": "Merge pull request #121 from harmoniqs/amico/s322-declared-loss-structure-hook\n\nGeneralize the KnotHVP capability into a declared loss structure (#120)",
          "timestamp": "2026-07-30T17:39:23-04:00",
          "tree_id": "55345abf26025c1a4e3e9215e85fe3e108a72e79",
          "url": "https://github.com/harmoniqs/DirectTrajOpt.jl/commit/0cd7923e79376e6bf26c3532246169addbc4a424"
        },
        "date": 1785450097378,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "xgate_convergence_ipopt_N51 [wall]",
            "value": 19.34668908,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [alloc]",
            "value": 4520098912,
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
            "value": 18.777289488,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [alloc]",
            "value": 4636043672,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [iters]",
            "value": 30,
            "unit": "iterations"
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
            "email": "47730232+aarontrowbridge@users.noreply.github.com",
            "name": "Aaron Trowbridge",
            "username": "aarontrowbridge"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "110708d9aac3d6ba771c21830cd926a21021e47c",
          "message": "fix(integrators): restore the multi-state BilinearIntegrator constructor (#139)\n\n* fix(integrators): restore the multi-state BilinearIntegrator constructor\n\nThe c9fdeb7 integrator refactor dropped the historical xs::AbstractVector{Symbol}\nconstructor, leaving every stacked-state caller — concretely Piccolo's exported\nVariationalKetIntegrator/VariationalUnitaryIntegrator (Piccolo #300) — with a\nMethodError on construction.\n\nAdditive restore following the codebase's existing multi-name convention\n(get_nonlinear_constraints already dispatches on x_names; Piccolissimo's\nexponential family carries it):\n\n- struct gains x_names::Vector{Symbol}; x_name::Symbol stays as the primary\n  (first) name, so existing field access keeps working\n- BilinearIntegrator(G, xs, u, traj) constructor; the single-name form\n  delegates via [x]\n- evaluate!/eval_jacobian/eval_hessian_of_lagrangian gather the stacked\n  state across all names (component ranges hoisted out of the ForwardDiff\n  closures)\n- get_nonlinear_constraints checks x_names before x_name so an integrator\n  carrying both fields sums the whole stack\n- test: split-component trajectory vs single-component reference — identical\n  residuals/Jacobians/Hessians on coinciding flat data + branch coverage\n\nFixes #138. Unblocks Piccolo #300 (variational integrator tests ride 2.0.3).\n\n* benchmark: pin HarmoniqsBenchmarks to v0.2.1 (DTO 0.10 compat + SolveStats)\n\nThe rev-pinned c38418c pin (compat DirectTrajOpt = 0.9 only) made both\nbenchmark CI suites unsatisfiable the moment DTO 0.10.0 hit General —\non this very PR, since its Project.toml declares 0.10. v0.2.1 widens\ncompat to 0.9/0.10 and wires the SolveStats return into\nbenchmark_solve!'s iteration counts (harmoniqs/HarmoniqsBenchmarks.jl#18).",
          "timestamp": "2026-08-21T18:55:43+02:00",
          "tree_id": "13b1ad8d8aa5ee30896b10f12cca37559439e2f0",
          "url": "https://github.com/harmoniqs/DirectTrajOpt.jl/commit/110708d9aac3d6ba771c21830cd926a21021e47c"
        },
        "date": 1787331696374,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "xgate_convergence_ipopt_N51 [wall]",
            "value": 20.002416119,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [alloc]",
            "value": 4439579480,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [iters]",
            "value": 20,
            "unit": "iterations"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [infidelity]",
            "value": 7.749301200732361e-11,
            "unit": "infidelity"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [wall]",
            "value": 19.270265802,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [alloc]",
            "value": 4131373488,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [iters]",
            "value": 23,
            "unit": "iterations"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [infidelity]",
            "value": 2.4343860260955807e-12,
            "unit": "infidelity"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "47730232+aarontrowbridge@users.noreply.github.com",
            "name": "Aaron Trowbridge",
            "username": "aarontrowbridge"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "082c1462def9a809ed8fec979872abdfe54c5604",
          "message": "release: v0.10.1 — multi-state BilinearIntegrator restore (#139) (#140)",
          "timestamp": "2026-08-21T19:30:26+02:00",
          "tree_id": "06d825b51ccbf1d029fa38dc6c616357c859dd9c",
          "url": "https://github.com/harmoniqs/DirectTrajOpt.jl/commit/082c1462def9a809ed8fec979872abdfe54c5604"
        },
        "date": 1787333528650,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "xgate_convergence_ipopt_N51 [wall]",
            "value": 16.579704121,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [alloc]",
            "value": 4439531144,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [iters]",
            "value": 20,
            "unit": "iterations"
          },
          {
            "name": "xgate_convergence_ipopt_N51 [infidelity]",
            "value": 7.749323405192854e-11,
            "unit": "infidelity"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [wall]",
            "value": 15.358710286,
            "unit": "s"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [alloc]",
            "value": 4131379744,
            "unit": "bytes"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [iters]",
            "value": 23,
            "unit": "iterations"
          },
          {
            "name": "xgate_convergence_madnlp_N51 [infidelity]",
            "value": 2.4334978476758806e-12,
            "unit": "infidelity"
          }
        ]
      }
    ]
  }
}