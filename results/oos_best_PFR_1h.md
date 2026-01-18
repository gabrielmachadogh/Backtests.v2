# OOS BEST (PFR 1h)

Split temporal (treino/teste). Thresholds calculados **só no treino**.

## Baseline (PFR cru)

| rr | trades_train | wr_train | trades_test | wr_test |
|---:|---:|---:|---:|---:|
| 1.0 | 335 | 49,0% | 84 | 44,0% |
| 1.5 | 322 | 41,6% | 81 | 34,6% |
| 2.0 | 312 | 34,9% | 78 | 30,8% |

## Top univariado (OOS)

| rr | feature | mode | pct | threshold | trades_test | wr_test | delta_test |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1.0 | ret_3_pct | high | 50 | 0.0350147 | 44 | 54,5% | 10,5% |
| 1.0 | vol_z | high | 50 | -0.237565 | 45 | 51,1% | 7,1% |
| 1.0 | ma_gap_pct | high | 60 | 1.26425 | 40 | 50,0% | 6,0% |
| 1.0 | body | high | 40 | 112.1 | 63 | 49,2% | 5,2% |
| 1.0 | vol_z | high | 60 | -0.422404 | 53 | 49,1% | 5,0% |
| 1.0 | bars_since_new_high | low | 60 | 4 | 49 | 49,0% | 4,9% |
| 1.0 | context_bars_since_extreme | low | 60 | 4 | 49 | 49,0% | 4,9% |
| 1.0 | body_pct | high | 60 | 0.180944 | 47 | 48,9% | 4,9% |
| 1.0 | ret_1_pct | high | 60 | 0.181272 | 47 | 48,9% | 4,9% |
| 1.0 | body | high | 50 | 80.3 | 72 | 48,6% | 4,6% |
| 1.0 | lower_wick | high | 40 | 144.58 | 58 | 48,3% | 4,2% |
| 1.0 | lower_wick_pct | low | 50 | 0.294918 | 54 | 48,1% | 4,1% |
| 1.0 | lower_wick | high | 60 | 82.96 | 73 | 47,9% | 3,9% |
| 1.0 | rsi | high | 60 | 60.3844 | 48 | 47,9% | 3,9% |
| 1.0 | body | high | 60 | 54.8 | 76 | 47,4% | 3,3% |
| 1.0 | lower_wick_pct | low | 40 | 0.240106 | 47 | 46,8% | 2,8% |
| 1.0 | pullback_from_new_high_atr | low | 50 | 1.044 | 47 | 46,8% | 2,8% |
| 1.0 | context_pullback_atr | low | 50 | 1.044 | 47 | 46,8% | 2,8% |
| 1.0 | slope_strength | high | 60 | 0.0998979 | 43 | 46,5% | 2,5% |
| 1.0 | ret_5_pct | high | 60 | 0.35575 | 43 | 46,5% | 2,5% |
| 1.0 | pullback_from_new_high_pct | low | 40 | 0.703209 | 54 | 46,3% | 2,2% |
| 1.0 | context_pullback_pct | low | 40 | 0.703209 | 54 | 46,3% | 2,2% |
| 1.0 | lower_wick | high | 50 | 111 | 67 | 46,3% | 2,2% |
| 1.0 | pullback_from_new_high_atr | high | 60 | 0.857903 | 46 | 45,7% | 1,6% |
| 1.0 | context_pullback_atr | high | 60 | 0.857903 | 46 | 45,7% | 1,6% |
| 1.0 | clv | high | 60 | 0.825125 | 55 | 45,5% | 1,4% |
| 1.0 | ret_3_pct | high | 60 | -0.0310817 | 55 | 45,5% | 1,4% |
| 1.0 | range | high | 50 | 269.5 | 75 | 45,3% | 1,3% |
| 1.0 | range | high | 40 | 352.7 | 64 | 45,3% | 1,3% |
| 1.0 | upper_wick | high | 50 | 32 | 62 | 45,2% | 1,1% |

## Top pairwise (OOS, grade fixa)

(sem pairwise)

