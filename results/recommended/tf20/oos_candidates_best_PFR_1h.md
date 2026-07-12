# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 41 | 48,8% | 3,7% | -0.0244 | 0.0745 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24529 | 47 | 46,8% | 1,8% | -0.0638 | 0.0351 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.271206 | 37 | 45,9% | 0,9% | -0.0811 | 0.0178 |
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 57 | 45,6% | 0,6% | -0.0877 | 0.0112 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23978 | 45 | 44,4% | 6,5% | 0.1111 | 0.1628 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.271314 | 36 | 44,4% | 6,5% | 0.1111 | 0.1628 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 39 | 41,0% | 3,1% | 0.0256 | 0.0774 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.280266 | 56 | 35,7% | -2,2% | -0.1071 | -0.0554 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9202 AND pullback_from_new_high_atr low50@1.05743 | 39 | 48,7% | 14,6% | 0.4615 | 0.4380 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 51 | 43,1% | 9,0% | 0.2941 | 0.2706 |
| RR2_rsi_high60 | 1 | rsi high60@59.9202 | 57 | 40,4% | 6,2% | 0.2105 | 0.1870 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 37 | 37,8% | 3,7% | 0.1351 | 0.1116 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 33,9% | -0,2% | 0.0179 | -0.0057 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 55 | 32,7% | -1,4% | -0.0182 | -0.0417 |

