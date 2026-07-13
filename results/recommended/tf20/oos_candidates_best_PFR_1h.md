# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 40 | 47,5% | 3,5% | -0.0500 | 0.0709 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.271349 | 37 | 45,9% | 2,0% | -0.0811 | 0.0398 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 46 | 45,7% | 1,7% | -0.0870 | 0.0339 |
| RR1_volz_high50 | 1 | vol_z high50@-0.259552 | 56 | 44,6% | 0,7% | -0.1071 | 0.0137 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.271278 | 35 | 45,7% | 7,8% | 0.1429 | 0.1946 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23498 | 45 | 44,4% | 6,5% | 0.1111 | 0.1628 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 38 | 42,1% | 4,2% | 0.0526 | 0.1044 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.277879 | 55 | 36,4% | -1,6% | -0.0909 | -0.0392 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8624 AND pullback_from_new_high_atr low50@1.05693 | 39 | 48,7% | 14,6% | 0.4615 | 0.4380 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 50 | 44,0% | 9,9% | 0.3200 | 0.2965 |
| RR2_rsi_high60 | 1 | rsi high60@59.8624 | 57 | 40,4% | 6,2% | 0.2105 | 0.1870 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 36 | 38,9% | 4,8% | 0.1667 | 0.1431 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 55 | 34,5% | 0,4% | 0.0364 | 0.0128 |
| RR2_volz_high50 | 1 | vol_z high50@-0.290896 | 54 | 33,3% | -0,8% | -0.0000 | -0.0235 |

