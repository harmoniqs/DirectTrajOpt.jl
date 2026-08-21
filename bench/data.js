window.BENCHMARK_DATA = {
  "lastUpdate": 1787332273708,
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
        "date": 1783028978105,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "bilinear_N51_ipopt [wall]",
            "value": 0.564104588,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_ipopt [alloc]",
            "value": 1432053240,
            "unit": "bytes"
          },
          {
            "name": "bilinear_N51_madnlp [wall]",
            "value": 0.363166886,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_madnlp [alloc]",
            "value": 980944936,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_ipopt [wall]",
            "value": 91.180284451,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_ipopt [alloc]",
            "value": 217045646104,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_madnlp [wall]",
            "value": 104.584031656,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_madnlp [alloc]",
            "value": 225218019568,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_ipopt [wall]",
            "value": 1.054411922,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_ipopt [alloc]",
            "value": 2467082832,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_madnlp [wall]",
            "value": 3.015062852,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_madnlp [alloc]",
            "value": 6876289696,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_ipopt [wall]",
            "value": 13.209394859,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_ipopt [alloc]",
            "value": 31199495256,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_madnlp [wall]",
            "value": 13.261389241,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_madnlp [alloc]",
            "value": 31834235888,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_ipopt [wall]",
            "value": 3.044575574,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_ipopt [alloc]",
            "value": 5664718144,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_madnlp [wall]",
            "value": 25.320407683,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_madnlp [alloc]",
            "value": 52473553120,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_ipopt [wall]",
            "value": 0.725991886,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_ipopt [alloc]",
            "value": 346568448,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_madnlp [wall]",
            "value": 0.799365378,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_madnlp [alloc]",
            "value": 1695380696,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_ipopt [wall]",
            "value": 0.012267772,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_ipopt [alloc]",
            "value": 22826992,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_madnlp [wall]",
            "value": 3.502803844,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_madnlp [alloc]",
            "value": 7970695944,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_ipopt [wall]",
            "value": 17.891640788,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_ipopt [alloc]",
            "value": 39827022448,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_madnlp [wall]",
            "value": 53.475827886,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_madnlp [alloc]",
            "value": 110802744736,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_ipopt [wall]",
            "value": 1.45122266,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_ipopt [alloc]",
            "value": 3282094576,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_madnlp [wall]",
            "value": 1.497549812,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_madnlp [alloc]",
            "value": 3376718344,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_ipopt [wall]",
            "value": 1.405805628,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_ipopt [alloc]",
            "value": 2731535776,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_madnlp [wall]",
            "value": 6.119426851,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_madnlp [alloc]",
            "value": 15372155472,
            "unit": "bytes"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_constraint [median]",
            "value": 869642,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_gradient [median]",
            "value": 223991,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_hessian_lagrangian [median]",
            "value": 19074658,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_jacobian [median]",
            "value": 1899791.5,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_objective [median]",
            "value": 200646,
            "unit": "ns"
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
        "date": 1787332272223,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "bilinear_N51_ipopt [wall]",
            "value": 1.491430216,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_ipopt [alloc]",
            "value": 4318456168,
            "unit": "bytes"
          },
          {
            "name": "bilinear_N51_ipopt [iters]",
            "value": 57,
            "unit": "iterations"
          },
          {
            "name": "bilinear_N51_madnlp [wall]",
            "value": 2.192956322,
            "unit": "s"
          },
          {
            "name": "bilinear_N51_madnlp [alloc]",
            "value": 6699092272,
            "unit": "bytes"
          },
          {
            "name": "bilinear_N51_madnlp [iters]",
            "value": 90,
            "unit": "iterations"
          },
          {
            "name": "scaling_N101_d16_ipopt [wall]",
            "value": 7.520233163,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_ipopt [alloc]",
            "value": 31020145144,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_ipopt [iters]",
            "value": 6,
            "unit": "iterations"
          },
          {
            "name": "scaling_N101_d16_madnlp [wall]",
            "value": 55.9360644,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d16_madnlp [alloc]",
            "value": 225313233696,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d16_madnlp [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N101_d4_ipopt [wall]",
            "value": 0.534550853,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_ipopt [alloc]",
            "value": 1722050512,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_ipopt [iters]",
            "value": 11,
            "unit": "iterations"
          },
          {
            "name": "scaling_N101_d4_madnlp [wall]",
            "value": 2.634601504,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d4_madnlp [alloc]",
            "value": 7053737504,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d4_madnlp [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N101_d8_ipopt [wall]",
            "value": 8.056191018,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_ipopt [alloc]",
            "value": 32208683712,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_ipopt [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N101_d8_madnlp [wall]",
            "value": 7.868935594,
            "unit": "s"
          },
          {
            "name": "scaling_N101_d8_madnlp [alloc]",
            "value": 31955404672,
            "unit": "bytes"
          },
          {
            "name": "scaling_N101_d8_madnlp [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N25_d16_ipopt [wall]",
            "value": 10.975087311,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_ipopt [alloc]",
            "value": 7626705880,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_ipopt [iters]",
            "value": 5,
            "unit": "iterations"
          },
          {
            "name": "scaling_N25_d16_madnlp [wall]",
            "value": 13.446405945,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d16_madnlp [alloc]",
            "value": 51756327144,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d16_madnlp [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N25_d4_ipopt [wall]",
            "value": 0.635619533,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_ipopt [alloc]",
            "value": 357071712,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_ipopt [iters]",
            "value": 0,
            "unit": "iterations"
          },
          {
            "name": "scaling_N25_d4_madnlp [wall]",
            "value": 2.685467847,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d4_madnlp [alloc]",
            "value": 2222879024,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d4_madnlp [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N25_d8_ipopt [wall]",
            "value": 0.007596808,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_ipopt [alloc]",
            "value": 22601216,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_ipopt [iters]",
            "value": 0,
            "unit": "iterations"
          },
          {
            "name": "scaling_N25_d8_madnlp [wall]",
            "value": 2.056089522,
            "unit": "s"
          },
          {
            "name": "scaling_N25_d8_madnlp [alloc]",
            "value": 8177629400,
            "unit": "bytes"
          },
          {
            "name": "scaling_N25_d8_madnlp [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N51_d16_ipopt [wall]",
            "value": 1.104382817,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_ipopt [alloc]",
            "value": 4518562000,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_ipopt [iters]",
            "value": 1,
            "unit": "iterations"
          },
          {
            "name": "scaling_N51_d16_madnlp [wall]",
            "value": 31.140133855,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d16_madnlp [alloc]",
            "value": 112532564928,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d16_madnlp [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N51_d4_ipopt [wall]",
            "value": 1.189099838,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_ipopt [alloc]",
            "value": 3434453600,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_ipopt [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N51_d4_madnlp [wall]",
            "value": 1.116697236,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d4_madnlp [alloc]",
            "value": 3321046616,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d4_madnlp [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "scaling_N51_d8_ipopt [wall]",
            "value": 2.296990438,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_ipopt [alloc]",
            "value": 9284381976,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_ipopt [iters]",
            "value": 28,
            "unit": "iterations"
          },
          {
            "name": "scaling_N51_d8_madnlp [wall]",
            "value": 3.835441453,
            "unit": "s"
          },
          {
            "name": "scaling_N51_d8_madnlp [alloc]",
            "value": 15722170280,
            "unit": "bytes"
          },
          {
            "name": "scaling_N51_d8_madnlp [iters]",
            "value": 50,
            "unit": "iterations"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_constraint [median]",
            "value": 850541,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_gradient [median]",
            "value": 219500,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_hessian_lagrangian [median]",
            "value": 16067535,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_jacobian [median]",
            "value": 1590312,
            "unit": "ns"
          },
          {
            "name": "evaluator_micro_bilinear_N51 / eval_objective [median]",
            "value": 189053,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}