window.BENCHMARK_DATA = {
  "lastUpdate": 1781069856467,
  "repoUrl": "https://github.com/harmoniqs/DirectTrajOpt.jl",
  "entries": {
    "DirectTrajOpt.jl benchmarks": [
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
        "date": 1780941848566,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "bilinear_N51_ipopt [wall]",
            "value": 0.718617798,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_ipopt [alloc]",
            "value": 1432053208,
            "unit": "bytes"
          },
          {
            "name": "bilinear_N51_madnlp [wall]",
            "value": 0.416276431,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_madnlp [alloc]",
            "value": 980944904,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_ipopt [wall]",
            "value": 108.851283831,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_ipopt [alloc]",
            "value": 211901588496,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_madnlp [wall]",
            "value": 114.005459223,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_madnlp [alloc]",
            "value": 229093097280,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_ipopt [wall]",
            "value": 3.984949246,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_ipopt [alloc]",
            "value": 7036030512,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_madnlp [wall]",
            "value": 3.710502609,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_madnlp [alloc]",
            "value": 6811996936,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_ipopt [wall]",
            "value": 0.999078163,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_ipopt [alloc]",
            "value": 2044325576,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_madnlp [wall]",
            "value": 16.049807659,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_madnlp [alloc]",
            "value": 30953807696,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_ipopt [wall]",
            "value": 9.076738626,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_ipopt [alloc]",
            "value": 3196544872,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_madnlp [wall]",
            "value": 27.121755285,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_madnlp [alloc]",
            "value": 51141705760,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_ipopt [wall]",
            "value": 0.650780342,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_ipopt [alloc]",
            "value": 346611520,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_madnlp [wall]",
            "value": 3.655652321,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_madnlp [alloc]",
            "value": 1896541688,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_ipopt [wall]",
            "value": 1.866167494,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_ipopt [alloc]",
            "value": 3358304792,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_madnlp [wall]",
            "value": 4.293217632,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_madnlp [alloc]",
            "value": 7970695912,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_ipopt [wall]",
            "value": 18.44591174,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_ipopt [alloc]",
            "value": 30901693552,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_madnlp [wall]",
            "value": 68.047828385,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_madnlp [alloc]",
            "value": 110806043800,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_ipopt [wall]",
            "value": 0.308899537,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_ipopt [alloc]",
            "value": 609791568,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_madnlp [wall]",
            "value": 1.836160408,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_madnlp [alloc]",
            "value": 3371633368,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_ipopt [wall]",
            "value": 0.147273036,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_ipopt [alloc]",
            "value": 358881312,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_madnlp [wall]",
            "value": 7.454427108,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_madnlp [alloc]",
            "value": 14829706416,
            "unit": "bytes"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_constraint [median]",
            "value": 859298,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_gradient [median]",
            "value": 233664.5,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_hessian_lagrangian [median]",
            "value": 26616012.5,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_jacobian [median]",
            "value": 2076219.5,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_objective [median]",
            "value": 206264.5,
            "unit": "ns"
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
        "date": 1780947051501,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "bilinear_N51_ipopt [wall]",
            "value": 0.474763568,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_ipopt [alloc]",
            "value": 1432053400,
            "unit": "bytes"
          },
          {
            "name": "bilinear_N51_madnlp [wall]",
            "value": 0.333557291,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_madnlp [alloc]",
            "value": 980944648,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_ipopt [wall]",
            "value": 95.92700406,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_ipopt [alloc]",
            "value": 213025011248,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_madnlp [wall]",
            "value": 111.241167852,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_madnlp [alloc]",
            "value": 229093097664,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_ipopt [wall]",
            "value": 2.1842982,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_ipopt [alloc]",
            "value": 5468986784,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_madnlp [wall]",
            "value": 2.64662755,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_madnlp [alloc]",
            "value": 6753287912,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_ipopt [wall]",
            "value": 0.81198524,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_ipopt [alloc]",
            "value": 2065909600,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_madnlp [wall]",
            "value": 12.089757477,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_madnlp [alloc]",
            "value": 31505978376,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_ipopt [wall]",
            "value": 6.319212076,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_ipopt [alloc]",
            "value": 2155582496,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_madnlp [wall]",
            "value": 25.675883233,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_madnlp [alloc]",
            "value": 52539124920,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_ipopt [wall]",
            "value": 0.015897833,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_ipopt [alloc]",
            "value": 38100888,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_madnlp [wall]",
            "value": 0.742593478,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_madnlp [alloc]",
            "value": 1695842504,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_ipopt [wall]",
            "value": 0.512803792,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_ipopt [alloc]",
            "value": 1263393120,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_madnlp [wall]",
            "value": 3.333935072,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_madnlp [alloc]",
            "value": 7970697448,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_ipopt [wall]",
            "value": 53.930286633,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_ipopt [alloc]",
            "value": 110805035840,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_madnlp [wall]",
            "value": 50.959850801,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_madnlp [alloc]",
            "value": 108755827064,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_ipopt [wall]",
            "value": 1.457487627,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_ipopt [alloc]",
            "value": 3274791008,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_madnlp [wall]",
            "value": 1.375473337,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_madnlp [alloc]",
            "value": 3294438984,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_ipopt [wall]",
            "value": 1.96417681,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_ipopt [alloc]",
            "value": 4679798544,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_madnlp [wall]",
            "value": 6.107400153,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_madnlp [alloc]",
            "value": 15508261776,
            "unit": "bytes"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_constraint [median]",
            "value": 667125,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_gradient [median]",
            "value": 178760,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_hessian_lagrangian [median]",
            "value": 17671070,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_jacobian [median]",
            "value": 1613754,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_objective [median]",
            "value": 154873,
            "unit": "ns"
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
        "date": 1781069854698,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "bilinear_N51_ipopt [wall]",
            "value": 0.594886648,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_ipopt [alloc]",
            "value": 1432053208,
            "unit": "bytes"
          },
          {
            "name": "bilinear_N51_madnlp [wall]",
            "value": 0.361800188,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_madnlp [alloc]",
            "value": 980944904,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_ipopt [wall]",
            "value": 93.672917389,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_ipopt [alloc]",
            "value": 210743355672,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_madnlp [wall]",
            "value": 107.24579776,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_madnlp [alloc]",
            "value": 229093097280,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_ipopt [wall]",
            "value": 3.401541932,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_ipopt [alloc]",
            "value": 7051499024,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_madnlp [wall]",
            "value": 3.367032081,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_madnlp [alloc]",
            "value": 6876295504,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_ipopt [wall]",
            "value": 16.128559269,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_ipopt [alloc]",
            "value": 33319877952,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_madnlp [wall]",
            "value": 14.095739864,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_madnlp [alloc]",
            "value": 31707271472,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_ipopt [wall]",
            "value": 8.347225608,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_ipopt [alloc]",
            "value": 2155582496,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_madnlp [wall]",
            "value": 26.215677905,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_madnlp [alloc]",
            "value": 52473553088,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_ipopt [wall]",
            "value": 1.00619233,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_ipopt [alloc]",
            "value": 347578416,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_madnlp [wall]",
            "value": 0.884202315,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_madnlp [alloc]",
            "value": 1695380664,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_ipopt [wall]",
            "value": 4.777003712,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_ipopt [alloc]",
            "value": 8426681296,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_madnlp [wall]",
            "value": 3.784919156,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_madnlp [alloc]",
            "value": 7800399808,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_ipopt [wall]",
            "value": 22.619076088,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_ipopt [alloc]",
            "value": 43996002616,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_madnlp [wall]",
            "value": 49.189627076,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_madnlp [alloc]",
            "value": 104626658368,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_ipopt [wall]",
            "value": 0.597400298,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_ipopt [alloc]",
            "value": 1253140304,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_madnlp [wall]",
            "value": 1.622207789,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_madnlp [alloc]",
            "value": 3294438600,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_ipopt [wall]",
            "value": 7.578442227,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_ipopt [alloc]",
            "value": 14648978096,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_madnlp [wall]",
            "value": 8.096213414,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_madnlp [alloc]",
            "value": 16411696264,
            "unit": "bytes"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_constraint [median]",
            "value": 850292,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_gradient [median]",
            "value": 236176,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_hessian_lagrangian [median]",
            "value": 21857630,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_jacobian [median]",
            "value": 2003085.5,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_objective [median]",
            "value": 206096,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}