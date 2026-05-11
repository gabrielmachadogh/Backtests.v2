# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 62 | 56,5% | 8,3% | 0.1290 | 0.1666 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.279665 | 55 | 56,4% | 8,2% | 0.1273 | 0.1649 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 78 | 55,1% | 7,0% | 0.1026 | 0.1402 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.29909 | 71 | 47,9% | -0,2% | -0.0423 | -0.0047 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0383034 AND slope_strength low80@0.278331 | 52 | 53,8% | 13,2% | 0.3462 | 0.3305 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 59 | 49,2% | 8,5% | 0.2288 | 0.2132 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.291583 | 78 | 46,2% | 5,5% | 0.1538 | 0.1382 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24324 | 71 | 45,1% | 4,4% | 0.1268 | 0.1111 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04913 | 68 | 44,1% | 8,6% | 0.3235 | 0.2590 |
| RR2_rsi_high60 | 1 | rsi high60@60.1055 | 75 | 44,0% | 8,5% | 0.3200 | 0.2555 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 57 | 43,9% | 8,4% | 0.3158 | 0.2513 |
| RR2_volz_high50 | 1 | vol_z high50@-0.320238 | 77 | 39,0% | 3,5% | 0.1688 | 0.1043 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 79 | 36,7% | 1,2% | 0.1013 | 0.0367 |

