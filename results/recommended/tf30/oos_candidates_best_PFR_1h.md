# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0372722 AND body_pct high60@0.178327 | 44 | 54,5% | 8,6% | 0.0909 | 0.1724 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 61 | 52,5% | 6,5% | 0.0492 | 0.1307 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.278776 | 54 | 51,9% | 5,9% | 0.0370 | 0.1185 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 81 | 50,6% | 4,7% | 0.0123 | 0.0938 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31381 | 68 | 45,6% | -0,3% | -0.0882 | -0.0068 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0393346 AND slope_strength low80@0.27717 | 51 | 49,0% | 10,6% | 0.2255 | 0.2640 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0393346 | 58 | 44,8% | 6,4% | 0.1207 | 0.1592 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24598 | 68 | 44,1% | 5,7% | 0.1029 | 0.1414 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.290896 | 81 | 42,0% | 3,5% | 0.0494 | 0.0878 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.1008 | 77 | 44,2% | 10,8% | 0.3247 | 0.3247 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 71 | 43,7% | 10,3% | 0.3099 | 0.3099 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 56 | 39,3% | 6,0% | 0.1786 | 0.1786 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 79 | 36,7% | 3,4% | 0.1013 | 0.1013 |
| RR2_volz_high50 | 1 | vol_z high50@-0.306383 | 78 | 34,6% | 1,3% | 0.0385 | 0.0385 |

