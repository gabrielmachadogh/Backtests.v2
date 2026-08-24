# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0371925 AND body_pct high60@0.178377 | 45 | 53,3% | 6,6% | 0.0667 | 0.1314 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.27717 | 56 | 51,8% | 5,0% | 0.0357 | 0.1005 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 64 | 51,6% | 4,8% | 0.0312 | 0.0960 |
| RR1_volz_high50 | 1 | vol_z high50@-0.280266 | 85 | 49,4% | 2,6% | -0.0118 | 0.0530 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31631 | 65 | 46,2% | -0,6% | -0.0769 | -0.0122 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.275429 | 53 | 49,1% | 9,5% | 0.2264 | 0.2376 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 61 | 44,3% | 4,7% | 0.1066 | 0.1178 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.3122 | 62 | 43,5% | 4,0% | 0.0887 | 0.0999 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.29227 | 85 | 41,2% | 1,6% | 0.0294 | 0.0406 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 74 | 44,6% | 10,0% | 0.3378 | 0.2994 |
| RR2_rsi_high60 | 1 | rsi high60@60.0535 | 79 | 44,3% | 9,7% | 0.3291 | 0.2907 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 60 | 40,0% | 5,4% | 0.2000 | 0.1615 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 82 | 37,8% | 3,2% | 0.1341 | 0.0957 |
| RR2_volz_high50 | 1 | vol_z high50@-0.319338 | 82 | 34,1% | -0,5% | 0.0244 | -0.0141 |

