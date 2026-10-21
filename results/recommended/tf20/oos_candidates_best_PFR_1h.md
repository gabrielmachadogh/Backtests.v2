# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23569 | 45 | 46,7% | 1,4% | -0.0667 | 0.0281 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 43 | 44,2% | -1,1% | -0.1163 | -0.0215 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.268975 | 39 | 43,6% | -1,7% | -0.1282 | -0.0335 |
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 58 | 43,1% | -2,2% | -0.1379 | -0.0432 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.21605 | 44 | 47,7% | 7,5% | 0.1932 | 0.1877 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.268507 | 38 | 39,5% | -0,7% | -0.0132 | -0.0186 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 42 | 38,1% | -2,1% | -0.0476 | -0.0531 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 60 | 36,7% | -3,6% | -0.0833 | -0.0888 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0091 AND pullback_from_new_high_atr low50@1.04658 | 41 | 46,3% | 11,5% | 0.3902 | 0.3453 |
| RR2_rsi_high60 | 1 | rsi high60@60.0091 | 57 | 42,1% | 7,3% | 0.2632 | 0.2182 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04658 | 51 | 41,2% | 6,3% | 0.2353 | 0.1904 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 54 | 37,0% | 2,2% | 0.1111 | 0.0662 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 41 | 34,1% | -0,7% | 0.0244 | -0.0206 |
| RR2_volz_high50 | 1 | vol_z high50@-0.285748 | 58 | 31,0% | -3,8% | -0.0690 | -0.1139 |

