# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.280777 | 56 | 60,7% | 11,9% | 0.2143 | 0.2375 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 63 | 60,3% | 11,5% | 0.2063 | 0.2296 |
| RR1_volz_high50 | 1 | vol_z high50@-0.290896 | 78 | 59,0% | 10,1% | 0.1795 | 0.2027 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.32407 | 62 | 48,4% | -0,5% | -0.0323 | -0.0090 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 60 | 51,7% | 11,3% | 0.2917 | 0.2836 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.29909 | 62 | 46,8% | 6,5% | 0.1694 | 0.1613 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.306383 | 77 | 46,8% | 6,4% | 0.1688 | 0.1608 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 58 | 43,1% | 9,8% | 0.2931 | 0.2931 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05792 | 69 | 42,0% | 8,7% | 0.2609 | 0.2609 |
| RR2_rsi_high60 | 1 | rsi high60@60.1445 | 70 | 41,4% | 8,1% | 0.2429 | 0.2429 |
| RR2_volz_high50 | 1 | vol_z high50@-0.333782 | 77 | 37,7% | 4,3% | 0.1299 | 0.1299 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 76 | 34,2% | 0,9% | 0.0263 | 0.0263 |

