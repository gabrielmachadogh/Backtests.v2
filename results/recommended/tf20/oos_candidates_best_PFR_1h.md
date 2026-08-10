# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 54 | 48,1% | 3,6% | -0.0370 | 0.0717 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 40 | 47,5% | 2,9% | -0.0500 | 0.0587 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24324 | 45 | 46,7% | 2,1% | -0.0667 | 0.0420 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.271171 | 37 | 45,9% | 1,4% | -0.0811 | 0.0276 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22997 | 43 | 46,5% | 8,3% | 0.1628 | 0.2077 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.270919 | 36 | 44,4% | 6,2% | 0.1111 | 0.1561 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 39 | 41,0% | 2,8% | 0.0256 | 0.0706 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 56 | 39,3% | 1,1% | -0.0179 | 0.0271 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9202 AND pullback_from_new_high_atr low50@1.04913 | 37 | 51,4% | 18,8% | 0.5405 | 0.5638 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04913 | 48 | 43,8% | 11,2% | 0.3125 | 0.3358 |
| RR2_rsi_high60 | 1 | rsi high60@59.9202 | 56 | 41,1% | 8,5% | 0.2321 | 0.2554 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 38 | 36,8% | 4,3% | 0.1053 | 0.1285 |
| RR2_volz_high50 | 1 | vol_z high50@-0.290896 | 56 | 33,9% | 1,4% | 0.0179 | 0.0411 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 33,9% | 1,4% | 0.0179 | 0.0411 |

