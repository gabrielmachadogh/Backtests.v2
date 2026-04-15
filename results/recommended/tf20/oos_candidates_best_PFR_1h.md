# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 52 | 50,0% | 4,5% | 0.0000 | 0.0909 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 40 | 50,0% | 4,5% | 0.0000 | 0.0909 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24094 | 49 | 46,9% | 1,5% | -0.0612 | 0.0297 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.271206 | 35 | 45,7% | 0,3% | -0.0857 | 0.0052 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22497 | 47 | 44,7% | 7,0% | 0.1170 | 0.1758 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 39 | 41,0% | 3,4% | 0.0256 | 0.0845 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.283478 | 52 | 38,5% | 0,8% | -0.0385 | 0.0204 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8238 AND pullback_from_new_high_atr low50@1.05743 | 38 | 39,5% | 7,8% | 0.1842 | 0.2330 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 38 | 36,8% | 5,1% | 0.1053 | 0.1540 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 49 | 36,7% | 5,0% | 0.1020 | 0.1508 |
| RR2_rsi_high60 | 1 | rsi high60@59.8238 | 55 | 36,4% | 4,7% | 0.0909 | 0.1397 |
| RR2_volz_high50 | 1 | vol_z high50@-0.292849 | 51 | 33,3% | 1,6% | -0.0000 | 0.0488 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 57 | 31,6% | -0,1% | -0.0526 | -0.0039 |

