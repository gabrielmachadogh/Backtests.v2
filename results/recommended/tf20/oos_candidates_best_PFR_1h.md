# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 45 | 46,7% | 1,9% | -0.0667 | 0.0375 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.267033 | 41 | 46,3% | 1,5% | -0.0732 | 0.0310 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23211 | 46 | 45,7% | 0,9% | -0.0870 | 0.0172 |
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 58 | 41,4% | -3,4% | -0.1724 | -0.0682 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.20293 | 46 | 45,7% | 3,3% | 0.1413 | 0.0815 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.266496 | 39 | 43,6% | 1,2% | 0.0897 | 0.0300 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 43 | 41,9% | -0,5% | 0.0465 | -0.0133 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.238376 | 55 | 38,2% | -4,2% | -0.0455 | -0.1052 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9202 AND pullback_from_new_high_atr low50@1.04443 | 44 | 45,5% | 8,8% | 0.3636 | 0.2636 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04443 | 52 | 44,2% | 7,6% | 0.3269 | 0.2269 |
| RR2_rsi_high60 | 1 | rsi high60@59.9202 | 61 | 41,0% | 4,3% | 0.2295 | 0.1295 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 54 | 40,7% | 4,1% | 0.2222 | 0.1222 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 42 | 38,1% | 1,4% | 0.1429 | 0.0429 |
| RR2_volz_high50 | 1 | vol_z high50@-0.269909 | 57 | 33,3% | -3,3% | -0.0000 | -0.1000 |

