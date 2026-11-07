# 2020: climate_vlm vs ClimODE (WeatherBench2 64x32, 179 ICs)

## RMSE

| var | model | 1 d | 3 d | 4 d | 5 d |
|---|---|---:|---:|---:|---:|
| z | climate_vlm | 574 | 919 | 980 | 1.02e+03 |
| z | persistence | 576 | 924 | 985 | 1.02e+03 |
| z | ClimODE | 225 | 509 | 637 | 782 |
| z | pgflow:seqfm_dct_pinn_blend_band | 144 | 353 | 451 | 536 |
| t | climate_vlm | 2.68 | 3.87 | 4.05 | 4.14 |
| t | persistence | 2.71 | 3.99 | 4.18 | 4.27 |
| t | ClimODE | 1.32 | 2.46 | 2.9 | 3.64 |
| t | pgflow:seqfm_dct_pinn_blend_band | 1.35 | 2.01 | 2.29 | 2.55 |
| t2m | climate_vlm | 2.22 | 2.94 | 3.09 | 3.19 |
| t2m | persistence | 1.74 | 2.65 | 2.81 | 2.93 |
| t2m | ClimODE | 2.73 | 4.41 | 3.96 | 6.27 |
| t2m | pgflow:seqfm_dct_pinn_blend_band | 1.17 | 1.68 | 1.88 | 2.06 |
| u10 | climate_vlm | 3.58 | 3.58 | 3.57 | 3.58 |
| u10 | persistence | 3.31 | 4.37 | 4.53 | 4.6 |
| u10 | ClimODE | 1.55 | 2.74 | 3.15 | 3.64 |
| u10 | pgflow:seqfm_dct_pinn_blend_band | 1.58 | 2.42 | 2.71 | 2.96 |
| v10 | climate_vlm | 3.74 | 3.74 | 3.73 | 3.74 |
| v10 | persistence | 4.06 | 4.94 | 5.02 | 5.04 |
| v10 | ClimODE | 1.6 | 2.98 | 3.48 | 4.21 |
| v10 | pgflow:seqfm_dct_pinn_blend_band | 1.69 | 2.56 | 2.88 | 3.15 |

## ACC

| var | model | 1 d | 3 d | 4 d | 5 d |
|---|---|---:|---:|---:|---:|
| z | climate_vlm | 0.836 | 0.576 | 0.518 | 0.485 |
| z | persistence | 0.837 | 0.577 | 0.519 | 0.486 |
| z | ClimODE | 0.976 | 0.872 | 0.799 | 0.700 |
| z | pgflow:seqfm_dct_pinn_blend_band | 0.990 | 0.937 | 0.892 | 0.843 |
| t | climate_vlm | 0.838 | 0.655 | 0.623 | 0.608 |
| t | persistence | 0.839 | 0.646 | 0.612 | 0.596 |
| t | ClimODE | 0.962 | 0.866 | 0.811 | 0.707 |
| t | pgflow:seqfm_dct_pinn_blend_band | 0.959 | 0.908 | 0.876 | 0.843 |
| t2m | climate_vlm | 0.905 | 0.829 | 0.812 | 0.800 |
| t2m | persistence | 0.941 | 0.865 | 0.848 | 0.836 |
| t2m | ClimODE | 0.856 | 0.639 | 0.690 | 0.403 |
| t2m | pgflow:seqfm_dct_pinn_blend_band | 0.974 | 0.945 | 0.930 | 0.915 |
| u10 | climate_vlm | 0.190 | 0.193 | 0.195 | 0.193 |
| u10 | persistence | 0.578 | 0.267 | 0.211 | 0.187 |
| u10 | ClimODE | 0.905 | 0.659 | 0.530 | 0.382 |
| u10 | pgflow:seqfm_dct_pinn_blend_band | 0.899 | 0.742 | 0.660 | 0.577 |
| v10 | climate_vlm | 0.113 | 0.119 | 0.118 | 0.119 |
| v10 | persistence | 0.398 | 0.115 | 0.085 | 0.078 |
| v10 | ClimODE | 0.903 | 0.627 | 0.469 | 0.270 |
| v10 | pgflow:seqfm_dct_pinn_blend_band | 0.890 | 0.724 | 0.630 | 0.533 |

## Gaussian CRPS (ClimODE min-max units)

| var | model | 1 d | 3 d | 4 d | 5 d |
|---|---|---:|---:|---:|---:|
| z | climate_vlm | 0.0192 | 0.0317 | 0.0341 | 0.0357 |
| z | persistence | 0.0262 | 0.0446 | 0.0481 | 0.0505 |
| z | ClimODE | 0.0092 | 0.0222 | 0.0312 | 0.0349 |
| z | pgflow:seqfm_dct_pinn_blend_band | 0.0064 | 0.0149 | 0.0186 | 0.0218 |
| t | climate_vlm | 0.0165 | 0.0242 | 0.0256 | 0.0264 |
| t | persistence | 0.0219 | 0.0344 | 0.0365 | 0.0377 |
| t | ClimODE | 0.0085 | 0.0166 | 0.0239 | 0.0256 |
| t | pgflow:seqfm_dct_pinn_blend_band | 0.0095 | 0.0143 | 0.0162 | 0.0179 |
| t2m | climate_vlm | 0.0105 | 0.0139 | 0.0147 | 0.0152 |
| t2m | persistence | 0.0103 | 0.0163 | 0.0175 | 0.0184 |
| t2m | ClimODE | 0.0095 | 0.0178 | 0.0208 | 0.0262 |
| t2m | pgflow:seqfm_dct_pinn_blend_band | 0.0059 | 0.0089 | 0.0100 | 0.0109 |
| u10 | climate_vlm | 0.0387 | 0.0388 | 0.0387 | 0.0389 |
| u10 | persistence | 0.0459 | 0.0631 | 0.0658 | 0.0673 |
| u10 | ClimODE | 0.0174 | 0.0319 | 0.0438 | 0.0435 |
| u10 | pgflow:seqfm_dct_pinn_blend_band | 0.0192 | 0.0288 | 0.0319 | 0.0346 |
| v10 | climate_vlm | 0.0391 | 0.0390 | 0.0390 | 0.0390 |
| v10 | persistence | 0.0520 | 0.0669 | 0.0685 | 0.0692 |
| v10 | ClimODE | 0.0175 | 0.0322 | 0.0451 | 0.0463 |
| v10 | pgflow:seqfm_dct_pinn_blend_band | 0.0197 | 0.0291 | 0.0324 | 0.0349 |

Units: z m2/s2, t K, t2m K, u10 m/s, v10 m/s.
truth coarsening RMSE (our HR truth averaged to 64x32 vs WB2 64x32), z500 / t2m at 24 h: 8.76 / 0.196.
Checks: {"persistence_max_rel_diff_vs_reference": 9.599185616455943e-07}
