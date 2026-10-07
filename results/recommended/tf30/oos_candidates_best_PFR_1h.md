# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0350147 AND body_pct high60@0.178278 | 48 | 54,2% | 5,9% | 0.0833 | 0.1188 |
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0350147 AND clv high60@0.828411 | 45 | 53,3% | 5,1% | 0.0667 | 0.1021 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 68 | 51,5% | 3,2% | 0.0294 | 0.0649 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.277751 | 59 | 50,8% | 2,6% | 0.0169 | 0.0524 |
| RR1_volz_high50 | 1 | vol_z high50@-0.280266 | 88 | 50,0% | 1,8% | 0.0000 | 0.0355 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.3156 | 66 | 48,5% | 0,3% | -0.0303 | 0.0052 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.28599 | 65 | 47,7% | 6,5% | 0.1923 | 0.1629 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.276009 | 56 | 46,4% | 5,3% | 0.1607 | 0.1313 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 65 | 43,1% | 1,9% | 0.0769 | 0.0475 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.29227 | 88 | 42,0% | 0,9% | 0.0511 | 0.0217 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05792 | 76 | 44,7% | 9,1% | 0.3421 | 0.2739 |
| RR2_rsi_high60 | 1 | rsi high60@59.9765 | 84 | 44,0% | 8,4% | 0.3214 | 0.2532 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 82 | 40,2% | 4,6% | 0.2073 | 0.1391 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 63 | 38,1% | 2,5% | 0.1429 | 0.0747 |
| RR2_volz_high50 | 1 | vol_z high50@-0.310795 | 84 | 34,5% | -1,1% | 0.0357 | -0.0325 |

