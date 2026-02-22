# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 46 | 54,3% | 9,1% | 0.0870 | 0.1822 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0338931 AND slope_strength low80@0.271349 | 40 | 52,5% | 7,3% | 0.0500 | 0.1452 |
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 46 | 52,2% | 6,9% | 0.0435 | 0.1387 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 40 | 50,0% | 4,8% | 0.0000 | 0.0952 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.271278 | 38 | 47,4% | 11,6% | 0.1842 | 0.2891 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24209 | 39 | 43,6% | 7,8% | 0.0897 | 0.1947 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 44 | 43,2% | 7,4% | 0.0795 | 0.1845 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.275492 | 48 | 37,5% | 1,7% | -0.0625 | 0.0424 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0298 | 47 | 40,4% | 10,0% | 0.2128 | 0.3014 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04786 | 44 | 38,6% | 8,3% | 0.1591 | 0.2477 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 43 | 37,2% | 6,8% | 0.1163 | 0.2049 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 51 | 33,3% | 3,0% | -0.0000 | 0.0886 |
| RR2_volz_high50 | 1 | vol_z high50@-0.288793 | 47 | 31,9% | 1,5% | -0.0426 | 0.0461 |

