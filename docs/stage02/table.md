Domain: East Asia 28N-51N 110E-143E; 724 test decisions (split D); lead 24 h.

## Routers on split D

| router | accuracy (D) | accuracy (C) | Brier | NLL | ECE | E / oracle | E / baseline | choices baseline / advection / moisture |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B_mlp | 0.575 | 0.539 | 0.504 | 0.763 | 0.034 | 1.096 | 0.433 | 0.01 / 0.62 / 0.37 |
| C0_laya_zeroshot | 0.022 | 0.015 | 0.706 | 1.156 | 0.353 | 2.532 | 1.000 | 1.00 / 0.00 / 0.00 |
| C1_laya_tuned | 0.613 | 0.548 | 0.495 | 0.758 | 0.035 | 1.074 | 0.424 | 0.00 / 1.00 / 0.00 |
| C2_laya_cached_head | 0.613 | 0.548 | 0.496 | 0.760 | 0.039 | 1.074 | 0.424 | 0.00 / 1.00 / 0.00 |

Majority class of split B on split D: accuracy 0.613. Best fixed expert on split B: advection.

## Routed fields on the domain (split D)

| method | E | E - baseline [95 % CI] | RMSE z500 | RMSE t850 | RMSE t2m | RMSE u10 | RMSE v10 | ACC z500 | ACC t850 | RMSE z500 - baseline [95 % CI] |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_baseline | 0.4081 | +0.0000 [+0.0000, +0.0000] | 484.7 | 2.821 | 2.351 | 2.574 | 2.846 | 0.6983 | 0.6932 | +0.00 [+0.00, +0.00] |
| A_advection | 0.1730 | -0.2351 [-0.2615, -0.2089] | 327.8 | 1.739 | 1.826 | 1.706 | 1.84 | 0.8402 | 0.8625 | -156.89 [-182.69, -131.07] |
| A_moisture | 0.1823 | -0.2258 [-0.2534, -0.1985] | 350.7 | 1.836 | 1.828 | 1.797 | 1.849 | 0.8179 | 0.8519 | -134.04 [-159.62, -108.68] |
| A_best_fixed | 0.1730 | -0.2351 [-0.2615, -0.2089] | 327.8 | 1.739 | 1.826 | 1.706 | 1.84 | 0.8402 | 0.8625 | -156.89 [-182.69, -131.07] |
| oracle | 0.1611 | -0.2469 [-0.2744, -0.2194] | 330.7 | 1.763 | 1.817 | 1.662 | 1.753 | 0.8387 | 0.8603 | -153.98 [-179.72, -128.57] |
| B_mlp_hard | 0.1766 | -0.2315 [-0.2574, -0.2052] | 338.6 | 1.79 | 1.836 | 1.74 | 1.834 | 0.8303 | 0.8557 | -146.12 [-170.90, -121.83] |
| B_mlp_soft | 0.1886 | -0.2195 [-0.2444, -0.1945] | 329 | 1.742 | 1.794 | 1.687 | 1.772 | 0.8387 | 0.8635 | -155.73 [-180.75, -131.09] |
| C0_laya_zeroshot_hard | 0.4081 | +0.0000 [+0.0000, +0.0000] | 484.7 | 2.821 | 2.351 | 2.574 | 2.846 | 0.6983 | 0.6932 | +0.00 [+0.00, +0.00] |
| C0_laya_zeroshot_soft | 0.2639 | -0.1442 [-0.1611, -0.1274] | 357.4 | 1.927 | 1.869 | 1.795 | 1.913 | 0.8135 | 0.8377 | -127.28 [-147.25, -107.13] |
| C1_laya_tuned_hard | 0.1730 | -0.2351 [-0.2615, -0.2089] | 327.8 | 1.739 | 1.826 | 1.706 | 1.84 | 0.8402 | 0.8625 | -156.89 [-182.69, -131.07] |
| C1_laya_tuned_soft | 0.1808 | -0.2273 [-0.2538, -0.2011] | 325.2 | 1.728 | 1.795 | 1.688 | 1.78 | 0.8418 | 0.8652 | -159.49 [-185.49, -134.05] |
| C2_laya_cached_head_hard | 0.1730 | -0.2351 [-0.2615, -0.2089] | 327.8 | 1.739 | 1.826 | 1.706 | 1.84 | 0.8402 | 0.8625 | -156.89 [-182.69, -131.07] |
| C2_laya_cached_head_soft | 0.1811 | -0.2270 [-0.2536, -0.2007] | 325.2 | 1.727 | 1.795 | 1.687 | 1.781 | 0.8419 | 0.8653 | -159.49 [-185.51, -134.07] |

## Labels

```
{
 "B": {
  "n": 1448,
  "share": {
   "baseline": 0.02,
   "advection": 0.584,
   "moisture": 0.396
  },
  "near_tie_share": 0.20096685082872928,
  "by_season": {
   "DJF": {
    "baseline": 0.04,
    "advection": 0.601,
    "moisture": 0.359
   },
   "JJA": {
    "baseline": 0.005,
    "advection": 0.66,
    "moisture": 0.334
   },
   "MAM": {
    "baseline": 0.027,
    "advection": 0.549,
    "moisture": 0.424
   },
   "SON": {
    "baseline": 0.008,
    "advection": 0.527,
    "moisture": 0.464
   }
  },
  "mean_E": {
   "baseline": 0.433,
   "advection": 0.19,
   "moisture": 0.1977
  }
 },
 "C": {
  "n": 722,
  "share": {
   "baseline": 0.015,
   "advection": 0.548,
   "moisture": 0.436
  },
  "near_tie_share": 0.18559556786703602,
  "by_season": {
   "DJF": {
    "baseline": 0.029,
    "advection": 0.465,
    "moisture": 0.506
   },
   "JJA": {
    "baseline": 0.016,
    "advection": 0.576,
    "moisture": 0.408
   },
   "MAM": {
    "baseline": 0.016,
    "advection": 0.549,
    "moisture": 0.435
   },
   "SON": {
    "baseline": 0.0,
    "advection": 0.599,
    "moisture": 0.401
   }
  },
  "mean_E": {
   "baseline": 0.4032,
   "advection": 0.1787,
   "moisture": 0.18
  }
 },
 "D": {
  "n": 724,
  "share": {
   "baseline": 0.022,
   "advection": 0.613,
   "moisture": 0.365
  },
  "near_tie_share": 0.18093922651933703,
  "by_season": {
   "DJF": {
    "baseline": 0.017,
    "advection": 0.626,
    "moisture": 0.356
   },
   "JJA": {
    "baseline": 0.0,
    "advection": 0.696,
    "moisture": 0.304
   },
   "MAM": {
    "baseline": 0.054,
    "advection": 0.582,
    "moisture": 0.364
   },
   "SON": {
    "baseline": 0.016,
    "advection": 0.549,
    "moisture": 0.434
   }
  },
  "mean_E": {
   "baseline": 0.4081,
   "advection": 0.173,
   "moisture": 0.1823
  }
 }
}
```

## Temperatures (split C) and cost

```
{
 "temperature": {
  "B_mlp": 2.9520253491306123,
  "C0_laya_zeroshot": 19.999999999999996,
  "C1_laya_tuned": 0.9272587164264275,
  "C2_laya_cached_head": 0.9751400075695753
 },
 "cost": {
  "B_mlp": {
   "sec_per_decision": 1.2332023878960916e-06,
   "train_sec": 0.9913363456726074,
   "params": 6915
  },
  "C0_laya_zeroshot": {
   "sec_per_decision": 0.2803974672525206,
   "peak_rss_gb": 3.136848
  },
  "C1_laya_tuned": {
   "sec_per_decision": 0.27724574897141535,
   "peak_rss_gb": 3.083484,
   "train_sec": 5232.7392246723175,
   "train_peak_rss_gb": 4.517524
  },
  "C2_laya_cached_head": {
   "sec_per_decision": 0.2803974672525206,
   "peak_rss_gb": 3.136848,
   "head_train_sec": 1.7269105911254883,
   "note": "decision cost = the frozen Laya pass that makes the option states"
  }
 }
}
```
