# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 46 | 54,3% | 9,6% | 0.0870 | 0.1928 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0338931 AND slope_strength low80@0.271349 | 40 | 52,5% | 7,8% | 0.0500 | 0.1559 |
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 46 | 52,2% | 7,5% | 0.0435 | 0.1494 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 41 | 48,8% | 4,1% | -0.0244 | 0.0815 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0338931 AND slope_strength low80@0.271242 | 38 | 47,4% | 11,6% | 0.1842 | 0.2891 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 44 | 43,2% | 7,4% | 0.0795 | 0.1845 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23863 | 40 | 42,5% | 6,7% | 0.0625 | 0.1674 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 47 | 38,3% | 2,5% | -0.0426 | 0.0624 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@59.9953 | 47 | 40,4% | 10,0% | 0.2128 | 0.3014 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04913 | 44 | 38,6% | 8,3% | 0.1591 | 0.2477 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0327714 | 43 | 37,2% | 6,8% | 0.1163 | 0.2049 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 50 | 34,0% | 3,6% | 0.0200 | 0.1086 |
| RR2_volz_high50 | 1 | vol_z high50@-0.286689 | 46 | 32,6% | 2,2% | -0.0217 | 0.0669 |

