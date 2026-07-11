# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 40 | 47,5% | 3,1% | -0.0500 | 0.0611 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24529 | 46 | 45,7% | 1,2% | -0.0870 | 0.0242 |
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 57 | 45,6% | 1,2% | -0.0877 | 0.0234 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.271206 | 36 | 44,4% | 0,0% | -0.1111 | 0.0000 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.271171 | 35 | 42,9% | 6,1% | 0.0714 | 0.1519 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23748 | 45 | 42,2% | 5,4% | 0.0556 | 0.1360 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 39 | 38,5% | 1,7% | -0.0385 | 0.0420 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.283478 | 57 | 35,1% | -1,7% | -0.1228 | -0.0423 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9202 AND pullback_from_new_high_atr low50@1.05743 | 39 | 48,7% | 15,4% | 0.4615 | 0.4615 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 51 | 43,1% | 9,8% | 0.2941 | 0.2941 |
| RR2_rsi_high60 | 1 | rsi high60@59.9202 | 57 | 40,4% | 7,0% | 0.2105 | 0.2105 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 36 | 36,1% | 2,8% | 0.0833 | 0.0833 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 33,9% | 0,6% | 0.0179 | 0.0179 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 55 | 32,7% | -0,6% | -0.0182 | -0.0182 |

