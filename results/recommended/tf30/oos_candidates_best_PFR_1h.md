# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 54,7% | 8,0% | 0.0938 | 0.1604 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.28011 | 57 | 54,4% | 7,7% | 0.0877 | 0.1544 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 81 | 51,9% | 5,2% | 0.0370 | 0.1037 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31327 | 68 | 45,6% | -1,1% | -0.0882 | -0.0216 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.278331 | 53 | 50,9% | 12,2% | 0.2736 | 0.3046 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 60 | 46,7% | 7,9% | 0.1667 | 0.1977 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 67 | 43,3% | 4,5% | 0.0821 | 0.1131 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.290896 | 80 | 42,5% | 3,7% | 0.0625 | 0.0935 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04913 | 69 | 43,5% | 10,1% | 0.3043 | 0.3043 |
| RR2_rsi_high60 | 1 | rsi high60@60.0961 | 77 | 42,9% | 9,5% | 0.2857 | 0.2857 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 58 | 41,4% | 8,0% | 0.2414 | 0.2414 |
| RR2_volz_high50 | 1 | vol_z high50@-0.319338 | 79 | 35,4% | 2,1% | 0.0633 | 0.0633 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 79 | 35,4% | 2,1% | 0.0633 | 0.0633 |

