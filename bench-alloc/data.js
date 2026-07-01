window.BENCHMARK_DATA = {
  "lastUpdate": 1782904414555,
  "repoUrl": "https://github.com/harmoniqs/DirectTrajOpt.jl",
  "entries": {
    "DirectTrajOpt.jl alloc profile": [
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
        "date": 1780951576666,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc_bilinear_N51_ipopt [alloc-total]",
            "value": 14441607,
            "unit": "bytes"
          },
          {
            "name": "alloc_bilinear_N51_ipopt [alloc-count]",
            "value": 26209,
            "unit": "allocs"
          },
          {
            "name": "alloc_bilinear_N51_madnlp [alloc-total]",
            "value": 9862528,
            "unit": "bytes"
          },
          {
            "name": "alloc_bilinear_N51_madnlp [alloc-count]",
            "value": 17530,
            "unit": "allocs"
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
        "date": 1781073890112,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc_bilinear_N51_ipopt [alloc-total]",
            "value": 14046504,
            "unit": "bytes"
          },
          {
            "name": "alloc_bilinear_N51_ipopt [alloc-count]",
            "value": 26307,
            "unit": "allocs"
          },
          {
            "name": "alloc_bilinear_N51_madnlp [alloc-total]",
            "value": 9266449,
            "unit": "bytes"
          },
          {
            "name": "alloc_bilinear_N51_madnlp [alloc-count]",
            "value": 17397,
            "unit": "allocs"
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
        "date": 1782904411341,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "alloc_bilinear_N51_ipopt [alloc-total]",
            "value": 13593046,
            "unit": "bytes"
          },
          {
            "name": "alloc_bilinear_N51_ipopt [alloc-count]",
            "value": 26115,
            "unit": "allocs"
          },
          {
            "name": "alloc_bilinear_N51_madnlp [alloc-total]",
            "value": 9980112,
            "unit": "bytes"
          },
          {
            "name": "alloc_bilinear_N51_madnlp [alloc-count]",
            "value": 17185,
            "unit": "allocs"
          }
        ]
      }
    ]
  }
}