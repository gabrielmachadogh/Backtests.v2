# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 63 | 54,0% | 7,7% | 0.0794 | 0.1540 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.280555 | 56 | 53,6% | 7,3% | 0.0714 | 0.1461 |
| RR1_volz_high50 | 1 | vol_z high50@-0.280266 | 81 | 51,9% | 5,6% | 0.0370 | 0.1117 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.3122 | 69 | 46,4% | 0,1% | -0.0725 | 0.0022 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.279221 | 53 | 50,9% | 12,2% | 0.2736 | 0.3046 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 60 | 46,7% | 7,9% | 0.1667 | 0.1977 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24392 | 70 | 44,3% | 5,5% | 0.1071 | 0.1382 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.29227 | 81 | 43,2% | 4,5% | 0.0802 | 0.1113 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0791 | 76 | 42,1% | 9,3% | 0.2632 | 0.2792 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 69 | 42,0% | 9,2% | 0.2609 | 0.2769 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 58 | 41,4% | 8,6% | 0.2414 | 0.2574 |
| RR2_volz_high50 | 1 | vol_z high50@-0.319788 | 79 | 35,4% | 2,6% | 0.0633 | 0.0793 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 78 | 34,6% | 1,8% | 0.0385 | 0.0545 |

