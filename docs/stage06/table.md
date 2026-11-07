# Stage 06 scores, 2020 (n = 356 issue times)

Accuracy / Brier skill score vs climatology (Brier summed over options) / ECE.

| method | drop_24 | z500_24 | coldest_24 | drop_72 | z500_72 | coldest_72 | mean |
|---|---|---|---|---|---|---|---|
| climatology (2016) | 0.81 / 0.00 / 0.00 | 0.32 / 0.00 / 0.03 | 0.35 / 0.00 / 0.01 | 0.82 / 0.00 / 0.02 | 0.30 / 0.00 / 0.04 | 0.33 / 0.00 / 0.00 | 0.49 / 0.00 / 0.02 |
| forecast read as written | 0.81 / -0.20 / 0.19 | 0.40 / -0.78 / 0.60 | 0.47 / -0.44 / 0.53 | 0.82 / -0.21 / 0.18 | 0.47 / -0.58 / 0.53 | 0.54 / -0.24 / 0.46 | 0.59 / -0.41 / 0.41 |
| logistic, forecast number | 0.81 / 0.09 / 0.04 | 0.38 / 0.02 / 0.04 | 0.60 / 0.26 / 0.07 | 0.83 / 0.15 / 0.03 | 0.52 / 0.11 / 0.06 | 0.58 / 0.24 / 0.06 | 0.62 / 0.15 / 0.05 |
| logistic, every number | 0.88 / 0.43 / 0.02 | 0.54 / 0.17 / 0.04 | 0.62 / 0.31 / 0.06 | 0.87 / 0.37 / 0.04 | 0.53 / 0.14 / 0.07 | 0.60 / 0.27 / 0.05 | 0.67 / 0.28 / 0.05 |
| logistic, every number, all 2016 | 0.91 / 0.51 / 0.05 | 0.54 / 0.19 / 0.04 | 0.64 / 0.33 / 0.07 | 0.89 / 0.40 / 0.06 | 0.52 / 0.15 / 0.08 | 0.60 / 0.28 / 0.09 | 0.68 / 0.31 / 0.06 |
| Laya zero-shot | 0.81 / -0.02 / 0.03 | 0.35 / -0.69 / 0.56 | 0.18 / -0.07 / 0.15 | 0.82 / -0.01 / 0.04 | 0.39 / -0.27 / 0.32 | 0.16 / -0.06 / 0.16 | 0.45 / -0.19 / 0.21 |
| Laya + calibration (2019) | 0.81 / -0.01 / 0.06 | 0.35 / 0.00 / 0.03 | 0.18 / -0.02 / 0.07 | 0.82 / -0.01 / 0.08 | 0.39 / 0.00 / 0.01 | 0.16 / -0.01 / 0.09 | 0.45 / -0.01 / 0.06 |
| Laya heads (2016) + calibration (2019) | 0.82 / 0.17 / 0.03 | 0.38 / 0.02 / 0.01 | 0.46 / 0.09 / 0.04 | 0.82 / -0.01 / 0.08 | 0.42 / 0.02 / 0.05 | 0.41 / 0.07 / 0.04 | 0.55 / 0.06 / 0.04 |

Laya sets and routes (coverage of the 90 % set / mean set size / share routed auto / accuracy when auto / readout):

| method | drop_24 | z500_24 | coldest_24 | drop_72 | z500_72 | coldest_72 |
|---|---|---|---|---|---|---|
| Laya zero-shot | 0.81 / 1.00 / 1.00 / 0.81 / engine | 0.35 / 1.00 / 0.90 / 0.35 / engine | 0.41 / 2.00 / 0.00 /  / engine | 0.82 / 1.00 / 1.00 / 0.82 / engine | 0.70 / 2.00 / 0.00 /  / engine | 0.43 / 2.00 / 0.00 /  / engine |
| Laya + calibration (2019) | 0.93 / 1.78 / 0.22 / 0.68 / engine | 0.94 / 2.70 / 0.00 /  / engine | 0.87 / 3.59 / 0.00 /  / engine | 0.93 / 1.75 / 0.25 / 0.72 / engine | 0.95 / 2.60 / 0.00 /  / engine | 0.88 / 3.68 / 0.00 /  / engine |
| Laya heads (2016) + calibration (2019) | 0.89 / 1.17 / 0.83 / 0.87 / ttt | 0.91 / 2.60 / 0.01 / 0.00 / ttt | 0.88 / 3.00 / 0.04 / 0.86 / ttt | 0.93 / 1.75 / 0.25 / 0.72 / engine | 0.92 / 2.70 / 0.01 / 0.67 / ttt | 0.90 / 3.13 / 0.02 / 0.88 / ttt |

Differences averaged over the questions (95 % interval, 7-day block bootstrap):

- laya_fit - lr_all: brier: +0.115 [+0.095, +0.137]
- laya_fit - lr_all: hit: -0.118 [-0.140, -0.097]
- laya_zero - rule: brier: -0.132 [-0.187, -0.075]
- laya_zero - rule: hit: -0.134 [-0.170, -0.098]
- laya_cal - laya_zero: brier: -0.121 [-0.148, -0.094]
- laya_cal - laya_zero: hit: +0.000 [+0.000, +0.000]
- laya_fit - laya_cal: brier: -0.036 [-0.050, -0.023]
- laya_fit - laya_cal: hit: +0.102 [+0.063, +0.142]
- lr_all - rule: brier: -0.405 [-0.462, -0.351]
- lr_all - rule: hit: +0.086 [+0.053, +0.119]
- lr_all - clim: brier: -0.147 [-0.178, -0.118]
- lr_all - clim: hit: +0.184 [+0.153, +0.217]

## Questions

- `drop_24` (noul): Will the 850 hPa temperature averaged over the domain be at least 0.9 K lower at +24 h than now?
- `z500_24` (score): How will the 500 hPa height averaged over the domain change from now to +24 h? Options: ['changes by less than -11 m', 'changes by -11 to +11 m', 'changes by more than +11 m']
- `coldest_24` (choice): Which quarter of the domain will have the largest 850 hPa temperature fall (or the smallest rise) from now to +24 h? Options: {'NW': 'the north-west quarter', 'NE': 'the north-east quarter', 'SW': 'the south-west quarter', 'SE': 'the south-east quarter'}
- `drop_72` (noul): Will the 850 hPa temperature averaged over the domain be at least 2.2 K lower at +72 h than now?
- `z500_72` (score): How will the 500 hPa height averaged over the domain change from now to +72 h? Options: ['changes by less than -25 m', 'changes by -25 to +24 m', 'changes by more than +24 m']
- `coldest_72` (choice): Which quarter of the domain will have the largest 850 hPa temperature fall (or the smallest rise) from now to +72 h? Options: {'NW': 'the north-west quarter', 'NE': 'the north-east quarter', 'SW': 'the south-west quarter', 'SE': 'the south-east quarter'}

## Examples

State: East Asia 28N-51N 110E-143E, issued 2020-01-05 00 UTC. Now: T850 -4.8 C (NW -6.4, NE -15.1, SW 2.6, SE -2.1), Z500 5555 m, 850 hPa temperature advection -1.6 K/day; past 24 h: T850 +1.4 K, Z500 +59 m. Model forecast, change from now to +24 h: T850 +0.7 K (NW -0.1, NE +2.1, SW -0.1, SE +1.0), Z500 -8 m. Model forecast, change from now to +72 h: T850 +1.1 K (NW -0.5, NE +3.8, SW -0.5, SE +1.7), Z500 -25 m.

| question | truth | climatology (2016) | forecast read as written | logistic, forecast number | logistic, every number | logistic, every number, all 2016 | Laya zero-shot | Laya + calibration (2019) | Laya heads (2016) + calibration (2019) |
|---|---|---|---|---|---|---|---|---|---|
| drop_24 | false | false (0.81) | false (1.00) | false (0.90) | false (1.00) | false (0.98) | false (0.85) | false (0.81) | false (0.86) |
| z500_24 | 2 | 2 (0.35) | 1 (1.00) | 0 (0.42) | 2 (0.76) | 2 (0.60) | 1 (0.92) | 1 (0.38) | 2 (0.38) |
| coldest_24 | NW | NW (0.34) | NW (1.00) | NW (0.52) | NW (0.72) | NW (0.80) | SW (0.32) | SW (0.25) | SW (0.48) |
| drop_72 | false | false (0.80) | false (1.00) | false (0.89) | true (0.59) | false (0.56) | false (0.88) | false (0.81) | false (0.81) |
| z500_72 | 0 | 2 (0.34) | 1 (1.00) | 0 (0.60) | 0 (0.79) | 0 (0.82) | 1 (0.74) | 1 (0.40) | 0 (0.37) |
| coldest_72 | NW | NW (0.33) | NW (1.00) | NW (0.55) | NW (0.72) | NW (0.65) | SW (0.32) | SW (0.25) | NW (0.40) |

State: East Asia 28N-51N 110E-143E, issued 2020-01-06 00 UTC. Now: T850 -3.4 C (NW -8.8, NE -11.4, SW 3.6, SE 1.2), Z500 5587 m, 850 hPa temperature advection +0.0 K/day; past 24 h: T850 +1.4 K, Z500 +32 m. Model forecast, change from now to +24 h: T850 +0.0 K (NW +0.4, NE +0.9, SW -0.7, SE -0.4), Z500 -13 m. Model forecast, change from now to +72 h: T850 +0.0 K (NW +0.8, NE +1.3, SW -1.3, SE -0.6), Z500 -35 m.

| question | truth | climatology (2016) | forecast read as written | logistic, forecast number | logistic, every number | logistic, every number, all 2016 | Laya zero-shot | Laya + calibration (2019) | Laya heads (2016) + calibration (2019) |
|---|---|---|---|---|---|---|---|---|---|
| drop_24 | false | false (0.81) | false (1.00) | false (0.77) | false (0.95) | false (0.73) | false (0.85) | false (0.81) | false (0.96) |
| z500_24 | 0 | 2 (0.35) | 0 (1.00) | 0 (0.44) | 0 (0.77) | 0 (0.68) | 1 (0.91) | 1 (0.38) | 2 (0.40) |
| coldest_24 | NW | NW (0.34) | SW (1.00) | SW (0.42) | NW (0.70) | NW (0.67) | SW (0.33) | SW (0.25) | SE (0.33) |
| drop_72 | false | false (0.80) | false (1.00) | false (0.77) | true (0.86) | true (0.73) | false (0.87) | false (0.80) | false (0.80) |
| z500_72 | 0 | 2 (0.34) | 0 (1.00) | 0 (0.66) | 0 (0.89) | 0 (0.93) | 1 (0.73) | 1 (0.39) | 2 (0.36) |
| coldest_72 | SW | NW (0.33) | SW (1.00) | SW (0.54) | NW (0.47) | SW (0.53) | SW (0.33) | SW (0.25) | NW (0.31) |

State: East Asia 28N-51N 110E-143E, issued 2020-01-07 00 UTC. Now: T850 -3.2 C (NW -11.9, NE -9.6, SW 1.7, SE 4.9), Z500 5580 m, 850 hPa temperature advection +0.2 K/day; past 24 h: T850 +0.2 K, Z500 -7 m. Model forecast, change from now to +24 h: T850 +0.1 K (NW +2.4, NE +0.1, SW +0.1, SE -2.0), Z500 -2 m. Model forecast, change from now to +72 h: T850 +0.0 K (NW +3.2, NE +0.3, SW +0.1, SE -3.0), Z500 -23 m.

| question | truth | climatology (2016) | forecast read as written | logistic, forecast number | logistic, every number | logistic, every number, all 2016 | Laya zero-shot | Laya + calibration (2019) | Laya heads (2016) + calibration (2019) |
|---|---|---|---|---|---|---|---|---|---|
| drop_24 | false | false (0.81) | false (1.00) | false (0.79) | false (0.81) | true (0.54) | false (0.86) | false (0.82) | false (0.98) |
| z500_24 | 0 | 2 (0.35) | 1 (1.00) | 0 (0.39) | 0 (0.88) | 0 (0.87) | 1 (0.92) | 1 (0.38) | 0 (0.43) |
| coldest_24 | SW | NW (0.34) | SE (1.00) | SW (0.35) | SW (0.55) | SW (0.55) | SW (0.32) | SW (0.25) | SW (0.40) |
| drop_72 | true | false (0.80) | false (1.00) | false (0.77) | true (0.59) | true (0.56) | false (0.87) | false (0.81) | false (0.81) |
| z500_72 | 0 | 2 (0.34) | 1 (1.00) | 0 (0.58) | 0 (0.74) | 0 (0.82) | 1 (0.72) | 1 (0.39) | 0 (0.40) |
| coldest_72 | SE | NW (0.33) | SE (1.00) | SE (0.54) | SE (0.40) | SE (0.63) | SW (0.32) | SW (0.25) | SE (0.30) |

