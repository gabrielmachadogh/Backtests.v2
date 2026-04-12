# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.280666 | 56 | 58,9% | 10,4% | 0.1786 | 0.2089 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 63 | 58,7% | 10,2% | 0.1746 | 0.2049 |
| RR1_volz_high50 | 1 | vol_z high50@-0.290896 | 80 | 56,2% | 7,8% | 0.1250 | 0.1553 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31488 | 67 | 46,3% | -2,2% | -0.0746 | -0.0443 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.279221 | 52 | 53,8% | 14,2% | 0.3462 | 0.3541 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 59 | 49,2% | 9,5% | 0.2288 | 0.2368 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.292849 | 77 | 45,5% | 5,8% | 0.1364 | 0.1443 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.25977 | 67 | 44,8% | 5,1% | 0.1194 | 0.1273 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 57 | 43,9% | 9,7% | 0.3158 | 0.2914 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 69 | 42,0% | 7,9% | 0.2609 | 0.2365 |
| RR2_rsi_high60 | 1 | rsi high60@60.1008 | 73 | 41,1% | 6,9% | 0.2329 | 0.2085 |
| RR2_volz_high50 | 1 | vol_z high50@-0.32701 | 78 | 37,2% | 3,0% | 0.1154 | 0.0910 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 78 | 34,6% | 0,5% | 0.0385 | 0.0141 |

