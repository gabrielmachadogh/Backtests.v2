# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0371925 AND body_pct high60@0.178797 | 45 | 53,3% | 7,0% | 0.0667 | 0.1391 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 64 | 51,6% | 5,2% | 0.0312 | 0.1037 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.27659 | 57 | 50,9% | 4,5% | 0.0175 | 0.0900 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 83 | 50,6% | 4,2% | 0.0120 | 0.0845 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31524 | 66 | 47,0% | 0,6% | -0.0606 | 0.0119 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.273797 | 54 | 48,1% | 8,8% | 0.2037 | 0.2189 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24598 | 66 | 45,5% | 6,1% | 0.1364 | 0.1515 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 61 | 44,3% | 4,9% | 0.1066 | 0.1217 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.291583 | 83 | 42,2% | 2,8% | 0.0542 | 0.0694 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0456 | 77 | 42,9% | 10,0% | 0.2857 | 0.3013 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 71 | 42,3% | 9,4% | 0.2676 | 0.2832 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 58 | 37,9% | 5,1% | 0.1379 | 0.1536 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 81 | 35,8% | 3,0% | 0.0741 | 0.0897 |
| RR2_volz_high50 | 1 | vol_z high50@-0.302112 | 79 | 32,9% | 0,1% | -0.0127 | 0.0030 |

