# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0350409 AND body_pct high60@0.178426 | 48 | 54,2% | 5,9% | 0.0833 | 0.1188 |
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0350409 AND clv high60@0.828244 | 45 | 53,3% | 5,1% | 0.0667 | 0.1021 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 67 | 52,2% | 4,0% | 0.0448 | 0.0802 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.278331 | 58 | 51,7% | 3,5% | 0.0345 | 0.0699 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 87 | 50,6% | 2,3% | 0.0115 | 0.0470 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31667 | 66 | 48,5% | 0,3% | -0.0303 | 0.0052 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.28599 | 65 | 47,7% | 6,2% | 0.1923 | 0.1553 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.276009 | 55 | 47,3% | 5,8% | 0.1818 | 0.1448 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 64 | 43,8% | 2,3% | 0.0938 | 0.0567 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.29227 | 87 | 42,5% | 1,0% | 0.0632 | 0.0262 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05792 | 75 | 45,3% | 9,5% | 0.3600 | 0.2837 |
| RR2_rsi_high60 | 1 | rsi high60@59.9765 | 83 | 44,6% | 8,7% | 0.3373 | 0.2610 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 82 | 40,2% | 4,4% | 0.2073 | 0.1310 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 62 | 38,7% | 2,8% | 0.1613 | 0.0850 |
| RR2_volz_high50 | 1 | vol_z high50@-0.310795 | 83 | 34,9% | -0,9% | 0.0482 | -0.0281 |

