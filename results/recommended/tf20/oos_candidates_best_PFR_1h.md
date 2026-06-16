# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 54 | 48,1% | 4,8% | -0.0370 | 0.0963 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 43 | 46,5% | 3,2% | -0.0698 | 0.0636 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271349 | 39 | 43,6% | 0,3% | -0.1282 | 0.0051 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 46 | 43,5% | 0,1% | -0.1304 | 0.0029 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.271278 | 38 | 42,1% | 7,2% | 0.0526 | 0.1805 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24209 | 44 | 38,6% | 3,8% | -0.0341 | 0.0938 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 42 | 38,1% | 3,2% | -0.0476 | 0.0803 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.275492 | 55 | 34,5% | -0,3% | -0.1364 | -0.0085 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9577 AND pullback_from_new_high_atr low50@1.05743 | 38 | 42,1% | 11,2% | 0.2632 | 0.3346 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 50 | 38,0% | 7,0% | 0.1400 | 0.2114 |
| RR2_rsi_high60 | 1 | rsi high60@59.9577 | 55 | 36,4% | 5,4% | 0.0909 | 0.1623 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 40 | 35,0% | 4,0% | 0.0500 | 0.1214 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 55 | 30,9% | -0,0% | -0.0727 | -0.0013 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 30,4% | -0,6% | -0.0893 | -0.0179 |

