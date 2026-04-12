# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 41 | 48,8% | 4,5% | -0.0244 | 0.0892 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 54 | 48,1% | 3,8% | -0.0370 | 0.0766 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.270416 | 35 | 45,7% | 1,4% | -0.0857 | 0.0279 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23863 | 49 | 44,9% | 0,6% | -0.1020 | 0.0116 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22497 | 46 | 43,5% | 6,6% | 0.0870 | 0.1643 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 39 | 41,0% | 4,1% | 0.0256 | 0.1030 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.283478 | 52 | 38,5% | 1,6% | -0.0385 | 0.0389 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.7769 AND pullback_from_new_high_atr low50@1.05693 | 37 | 37,8% | 6,1% | 0.1351 | 0.1839 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 38 | 36,8% | 5,1% | 0.1053 | 0.1540 |
| RR2_rsi_high60 | 1 | rsi high60@59.7769 | 55 | 36,4% | 4,7% | 0.0909 | 0.1397 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 48 | 35,4% | 3,7% | 0.0625 | 0.1113 |
| RR2_volz_high50 | 1 | vol_z high50@-0.29227 | 51 | 33,3% | 1,6% | -0.0000 | 0.0488 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 57 | 31,6% | -0,1% | -0.0526 | -0.0039 |

