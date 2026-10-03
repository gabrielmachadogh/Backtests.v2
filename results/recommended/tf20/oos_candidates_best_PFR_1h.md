# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.239186 | 57 | 49,1% | 2,3% | -0.0175 | 0.0463 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23569 | 48 | 47,9% | 1,1% | -0.0417 | 0.0222 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 42 | 47,6% | 0,8% | -0.0476 | 0.0162 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.270667 | 38 | 47,4% | 0,6% | -0.0526 | 0.0112 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22051 | 46 | 47,8% | 7,8% | 0.1957 | 0.1957 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.270164 | 36 | 41,7% | 1,7% | 0.0417 | 0.0417 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 40 | 40,0% | 0,0% | 0.0000 | 0.0000 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 58 | 39,7% | -0,3% | -0.0086 | -0.0086 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0456 AND pullback_from_new_high_atr low50@1.04572 | 37 | 51,4% | 16,1% | 0.5405 | 0.4837 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04572 | 49 | 44,9% | 9,7% | 0.3469 | 0.2901 |
| RR2_rsi_high60 | 1 | rsi high60@60.0456 | 55 | 43,6% | 8,4% | 0.3091 | 0.2523 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 55 | 36,4% | 1,1% | 0.0909 | 0.0341 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 39 | 35,9% | 0,7% | 0.0769 | 0.0201 |
| RR2_volz_high50 | 1 | vol_z high50@-0.286218 | 57 | 33,3% | -1,9% | -0.0000 | -0.0568 |

