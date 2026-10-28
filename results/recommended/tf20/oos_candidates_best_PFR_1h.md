# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 45 | 46,7% | 1,9% | -0.0667 | 0.0375 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.267569 | 41 | 46,3% | 1,5% | -0.0732 | 0.0310 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.22854 | 46 | 45,7% | 0,9% | -0.0870 | 0.0172 |
| RR1_volz_high50 | 1 | vol_z high50@-0.237565 | 57 | 42,1% | -2,7% | -0.1579 | -0.0537 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.20803 | 44 | 47,7% | 5,3% | 0.1932 | 0.1334 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.267033 | 40 | 42,5% | 0,1% | 0.0625 | 0.0027 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 44 | 40,9% | -1,5% | 0.0227 | -0.0371 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.239186 | 55 | 38,2% | -4,2% | -0.0455 | -0.1052 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9202 AND pullback_from_new_high_atr low50@1.04443 | 43 | 46,5% | 9,4% | 0.3953 | 0.2830 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04443 | 51 | 45,1% | 8,0% | 0.3529 | 0.2406 |
| RR2_rsi_high60 | 1 | rsi high60@59.9202 | 60 | 41,7% | 4,6% | 0.2500 | 0.1376 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 53 | 41,5% | 4,4% | 0.2453 | 0.1329 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 42 | 38,1% | 1,0% | 0.1429 | 0.0305 |
| RR2_volz_high50 | 1 | vol_z high50@-0.269909 | 56 | 33,9% | -3,2% | 0.0179 | -0.0945 |

