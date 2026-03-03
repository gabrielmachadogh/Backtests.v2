# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.28111 | 56 | 60,7% | 11,9% | 0.2143 | 0.2379 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 63 | 60,3% | 11,5% | 0.2063 | 0.2300 |
| RR1_volz_high50 | 1 | vol_z high50@-0.291583 | 77 | 58,4% | 9,6% | 0.1688 | 0.1925 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31667 | 61 | 49,2% | 0,4% | -0.0164 | 0.0072 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 61 | 50,8% | 11,0% | 0.2705 | 0.2746 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.25977 | 62 | 46,8% | 6,9% | 0.1694 | 0.1734 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.306383 | 76 | 46,1% | 6,2% | 0.1513 | 0.1554 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 59 | 42,4% | 8,8% | 0.2712 | 0.2628 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05792 | 67 | 41,8% | 8,2% | 0.2537 | 0.2453 |
| RR2_rsi_high60 | 1 | rsi high60@60.1942 | 66 | 40,9% | 7,3% | 0.2273 | 0.2189 |
| RR2_volz_high50 | 1 | vol_z high50@-0.333782 | 76 | 38,2% | 4,5% | 0.1447 | 0.1363 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 75 | 33,3% | -0,3% | -0.0000 | -0.0084 |

