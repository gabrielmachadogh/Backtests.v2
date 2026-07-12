# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0372722 AND body_pct high60@0.178327 | 44 | 54,5% | 8,2% | 0.0909 | 0.1644 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 62 | 53,2% | 6,9% | 0.0645 | 0.1380 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.278776 | 55 | 52,7% | 6,4% | 0.0545 | 0.1281 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 81 | 50,6% | 4,3% | 0.0123 | 0.0859 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31381 | 69 | 46,4% | 0,1% | -0.0725 | 0.0011 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0393346 AND slope_strength low80@0.27717 | 52 | 50,0% | 11,1% | 0.2500 | 0.2767 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0393346 | 59 | 45,8% | 6,8% | 0.1441 | 0.1708 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24598 | 69 | 44,9% | 6,0% | 0.1232 | 0.1499 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.290896 | 81 | 42,0% | 3,0% | 0.0494 | 0.0761 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.1008 | 77 | 44,2% | 10,3% | 0.3247 | 0.3089 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 71 | 43,7% | 9,8% | 0.3099 | 0.2941 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 57 | 40,4% | 6,5% | 0.2105 | 0.1948 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 79 | 36,7% | 2,9% | 0.1013 | 0.0855 |
| RR2_volz_high50 | 1 | vol_z high50@-0.306383 | 78 | 34,6% | 0,8% | 0.0385 | 0.0227 |

