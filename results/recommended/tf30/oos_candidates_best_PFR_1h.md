# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 63 | 54,0% | 7,3% | 0.0794 | 0.1460 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.279665 | 56 | 53,6% | 6,9% | 0.0714 | 0.1381 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 81 | 51,9% | 5,2% | 0.0370 | 0.1037 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31434 | 68 | 45,6% | -1,1% | -0.0882 | -0.0216 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.278331 | 53 | 50,9% | 11,7% | 0.2736 | 0.2928 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 60 | 46,7% | 7,4% | 0.1667 | 0.1859 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 68 | 44,1% | 4,9% | 0.1029 | 0.1222 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.290896 | 81 | 43,2% | 4,0% | 0.0802 | 0.0995 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 71 | 43,7% | 10,3% | 0.3099 | 0.3099 |
| RR2_rsi_high60 | 1 | rsi high60@60.0706 | 78 | 43,6% | 10,3% | 0.3077 | 0.3077 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 57 | 40,4% | 7,0% | 0.2105 | 0.2105 |
| RR2_volz_high50 | 1 | vol_z high50@-0.306383 | 78 | 35,9% | 2,6% | 0.0769 | 0.0769 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 79 | 35,4% | 2,1% | 0.0633 | 0.0633 |

