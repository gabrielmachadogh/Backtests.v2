# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.280666 | 55 | 61,8% | 12,2% | 0.2364 | 0.2441 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 62 | 61,3% | 11,7% | 0.2258 | 0.2336 |
| RR1_volz_high50 | 1 | vol_z high50@-0.291583 | 78 | 59,0% | 9,4% | 0.1795 | 0.1872 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31971 | 62 | 48,4% | -1,2% | -0.0323 | -0.0245 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.279221 | 53 | 56,6% | 16,3% | 0.4151 | 0.4070 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 60 | 51,7% | 11,3% | 0.2917 | 0.2836 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.25977 | 64 | 46,9% | 6,6% | 0.1719 | 0.1638 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.293428 | 76 | 46,1% | 5,7% | 0.1513 | 0.1433 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 58 | 43,1% | 8,9% | 0.2931 | 0.2681 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 68 | 42,6% | 8,5% | 0.2794 | 0.2544 |
| RR2_rsi_high60 | 1 | rsi high60@60.1102 | 70 | 41,4% | 7,3% | 0.2429 | 0.2179 |
| RR2_volz_high50 | 1 | vol_z high50@-0.338748 | 77 | 37,7% | 3,5% | 0.1299 | 0.1049 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 75 | 34,7% | 0,5% | 0.0400 | 0.0150 |

