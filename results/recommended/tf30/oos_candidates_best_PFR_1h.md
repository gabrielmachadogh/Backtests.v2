# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 63 | 55,6% | 8,5% | 0.1111 | 0.1708 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.280666 | 56 | 55,4% | 8,3% | 0.1071 | 0.1668 |
| RR1_volz_high50 | 1 | vol_z high50@-0.285581 | 81 | 53,1% | 6,1% | 0.0617 | 0.1214 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31381 | 68 | 47,1% | 0,0% | -0.0588 | 0.0009 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.279665 | 53 | 52,8% | 13,3% | 0.3208 | 0.3324 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 60 | 48,3% | 8,8% | 0.2083 | 0.2200 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24598 | 68 | 45,6% | 6,1% | 0.1397 | 0.1513 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.292849 | 81 | 44,4% | 4,9% | 0.1111 | 0.1227 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.062 | 76 | 43,4% | 9,8% | 0.3026 | 0.2946 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04913 | 68 | 42,6% | 9,0% | 0.2794 | 0.2714 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 57 | 42,1% | 8,5% | 0.2632 | 0.2552 |
| RR2_volz_high50 | 1 | vol_z high50@-0.320238 | 79 | 36,7% | 3,1% | 0.1013 | 0.0933 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 79 | 35,4% | 1,8% | 0.0633 | 0.0553 |

