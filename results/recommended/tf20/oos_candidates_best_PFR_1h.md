# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 55 | 47,3% | 3,2% | -0.0545 | 0.0637 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.271206 | 37 | 45,9% | 1,9% | -0.0811 | 0.0372 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24094 | 46 | 45,7% | 1,6% | -0.0870 | 0.0313 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 40 | 45,0% | 0,9% | -0.1000 | 0.0183 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22497 | 44 | 45,5% | 6,6% | 0.1364 | 0.1641 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.271171 | 36 | 44,4% | 5,6% | 0.1111 | 0.1389 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 39 | 41,0% | 2,1% | 0.0256 | 0.0534 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 57 | 40,4% | 1,5% | 0.0088 | 0.0365 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0377 AND pullback_from_new_high_atr low50@1.04658 | 37 | 51,4% | 16,9% | 0.5405 | 0.5061 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04658 | 49 | 44,9% | 10,4% | 0.3469 | 0.3125 |
| RR2_rsi_high60 | 1 | rsi high60@60.0377 | 53 | 43,4% | 8,9% | 0.3019 | 0.2674 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 38 | 36,8% | 2,4% | 0.1053 | 0.0708 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 57 | 35,1% | 0,6% | 0.0526 | 0.0181 |
| RR2_volz_high50 | 1 | vol_z high50@-0.286689 | 56 | 33,9% | -0,6% | 0.0179 | -0.0166 |

