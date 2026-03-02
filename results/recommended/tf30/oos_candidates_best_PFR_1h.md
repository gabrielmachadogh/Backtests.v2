# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.281728 | 56 | 60,7% | 11,9% | 0.2143 | 0.2379 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 63 | 60,3% | 11,5% | 0.2063 | 0.2300 |
| RR1_volz_high50 | 1 | vol_z high50@-0.290896 | 76 | 59,2% | 10,4% | 0.1842 | 0.2078 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31971 | 60 | 50,0% | 1,2% | 0.0000 | 0.0236 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 61 | 50,8% | 10,7% | 0.2705 | 0.2664 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.25977 | 61 | 47,5% | 7,4% | 0.1885 | 0.1844 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.306383 | 75 | 46,7% | 6,5% | 0.1667 | 0.1626 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 59 | 42,4% | 8,5% | 0.2712 | 0.2542 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05792 | 67 | 41,8% | 7,9% | 0.2537 | 0.2368 |
| RR2_rsi_high60 | 1 | rsi high60@60.1942 | 65 | 41,5% | 7,6% | 0.2462 | 0.2292 |
| RR2_volz_high50 | 1 | vol_z high50@-0.333782 | 75 | 38,7% | 4,8% | 0.1600 | 0.1431 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 74 | 33,8% | -0,1% | 0.0135 | -0.0034 |

