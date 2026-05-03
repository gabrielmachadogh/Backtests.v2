# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 61 | 55,7% | 8,0% | 0.1148 | 0.1602 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.279665 | 54 | 55,6% | 7,8% | 0.1111 | 0.1566 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 77 | 54,5% | 6,8% | 0.0909 | 0.1364 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.29909 | 71 | 47,9% | 0,2% | -0.0423 | 0.0032 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0383034 AND slope_strength low80@0.278331 | 51 | 52,9% | 12,8% | 0.3235 | 0.3196 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 58 | 48,3% | 8,1% | 0.2069 | 0.2030 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.291583 | 77 | 45,5% | 5,3% | 0.1364 | 0.1324 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24324 | 71 | 45,1% | 4,9% | 0.1268 | 0.1228 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04913 | 68 | 44,1% | 9,2% | 0.3235 | 0.2747 |
| RR2_rsi_high60 | 1 | rsi high60@60.1055 | 74 | 43,2% | 8,3% | 0.2973 | 0.2485 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 56 | 42,9% | 7,9% | 0.2857 | 0.2369 |
| RR2_volz_high50 | 1 | vol_z high50@-0.320238 | 76 | 38,2% | 3,2% | 0.1447 | 0.0960 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 79 | 36,7% | 1,7% | 0.1013 | 0.0525 |

