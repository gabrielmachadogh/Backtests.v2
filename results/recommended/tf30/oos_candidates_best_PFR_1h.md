# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.280555 | 56 | 60,7% | 11,5% | 0.2143 | 0.2297 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 63 | 60,3% | 11,1% | 0.2063 | 0.2217 |
| RR1_volz_high50 | 1 | vol_z high50@-0.290896 | 78 | 57,7% | 8,5% | 0.1538 | 0.1692 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31667 | 63 | 47,6% | -1,6% | -0.0476 | -0.0322 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.279221 | 53 | 56,6% | 16,6% | 0.4151 | 0.4151 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 60 | 51,7% | 11,7% | 0.2917 | 0.2917 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.25977 | 64 | 46,9% | 6,9% | 0.1719 | 0.1719 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.293428 | 76 | 46,1% | 6,1% | 0.1513 | 0.1513 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 58 | 43,1% | 9,2% | 0.2931 | 0.2766 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 69 | 42,0% | 8,1% | 0.2609 | 0.2443 |
| RR2_rsi_high60 | 1 | rsi high60@60.1102 | 71 | 40,8% | 7,0% | 0.2254 | 0.2088 |
| RR2_volz_high50 | 1 | vol_z high50@-0.338748 | 77 | 37,7% | 3,8% | 0.1299 | 0.1133 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 76 | 34,2% | 0,3% | 0.0263 | 0.0098 |

