# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.280666 | 56 | 58,9% | 10,1% | 0.1786 | 0.2015 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 63 | 58,7% | 9,9% | 0.1746 | 0.1975 |
| RR1_volz_high50 | 1 | vol_z high50@-0.290896 | 79 | 57,0% | 8,1% | 0.1392 | 0.1621 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31488 | 66 | 47,0% | -1,9% | -0.0606 | -0.0377 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.279665 | 53 | 54,7% | 14,2% | 0.3679 | 0.3560 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 60 | 50,0% | 9,5% | 0.2500 | 0.2381 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.293428 | 77 | 46,8% | 6,3% | 0.1688 | 0.1569 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24598 | 67 | 46,3% | 5,8% | 0.1567 | 0.1448 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 57 | 43,9% | 9,4% | 0.3158 | 0.2830 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 69 | 42,0% | 7,6% | 0.2609 | 0.2281 |
| RR2_rsi_high60 | 1 | rsi high60@60.1008 | 72 | 41,7% | 7,2% | 0.2500 | 0.2172 |
| RR2_volz_high50 | 1 | vol_z high50@-0.32701 | 77 | 37,7% | 3,2% | 0.1299 | 0.0971 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 77 | 35,1% | 0,6% | 0.0519 | 0.0192 |

