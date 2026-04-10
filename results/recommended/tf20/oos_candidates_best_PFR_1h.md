# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 52 | 50,0% | 5,7% | 0.0000 | 0.1136 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 41 | 48,8% | 4,5% | -0.0244 | 0.0892 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.270667 | 35 | 45,7% | 1,4% | -0.0857 | 0.0279 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23569 | 49 | 44,9% | 0,6% | -0.1020 | 0.0116 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22051 | 47 | 44,7% | 7,8% | 0.1170 | 0.1944 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 40 | 40,0% | 3,1% | 0.0000 | 0.0774 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.286689 | 53 | 39,6% | 2,7% | -0.0094 | 0.0679 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.73 AND pullback_from_new_high_atr low50@1.05743 | 39 | 38,5% | 6,8% | 0.1538 | 0.2026 |
| RR2_rsi_high60 | 1 | rsi high60@59.73 | 57 | 36,8% | 5,1% | 0.1053 | 0.1540 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 39 | 35,9% | 4,2% | 0.0769 | 0.1257 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 49 | 34,7% | 3,0% | 0.0408 | 0.0896 |
| RR2_volz_high50 | 1 | vol_z high50@-0.292849 | 51 | 33,3% | 1,6% | -0.0000 | 0.0488 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 32,1% | 0,4% | -0.0357 | 0.0131 |

