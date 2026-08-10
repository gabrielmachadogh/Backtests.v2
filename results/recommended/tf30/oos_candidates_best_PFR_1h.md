# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0361298 AND body_pct high60@0.178327 | 45 | 53,3% | 7,0% | 0.0667 | 0.1391 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 51,6% | 5,2% | 0.0312 | 0.1037 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.276009 | 57 | 50,9% | 4,5% | 0.0175 | 0.0900 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 84 | 50,0% | 3,6% | 0.0000 | 0.0725 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31381 | 66 | 47,0% | 0,6% | -0.0606 | 0.0119 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.273797 | 54 | 48,1% | 9,1% | 0.2037 | 0.2263 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24598 | 66 | 45,5% | 6,4% | 0.1364 | 0.1589 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 61 | 44,3% | 5,2% | 0.1066 | 0.1291 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.291583 | 84 | 41,7% | 2,6% | 0.0417 | 0.0642 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0456 | 78 | 43,6% | 10,3% | 0.3077 | 0.3077 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 72 | 43,1% | 9,7% | 0.2917 | 0.2917 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 59 | 39,0% | 5,6% | 0.1695 | 0.1695 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 82 | 36,6% | 3,3% | 0.0976 | 0.0976 |
| RR2_volz_high50 | 1 | vol_z high50@-0.302112 | 80 | 33,8% | 0,4% | 0.0125 | 0.0125 |

