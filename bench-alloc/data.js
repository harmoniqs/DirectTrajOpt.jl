window.BENCHMARK_DATA = {
  "lastUpdate": 1780951579404,
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
      }
    ]
  }
}