# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0372324 AND body_pct high60@0.178229 | 45 | 55,6% | 9,6% | 0.1111 | 0.1926 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 62 | 53,2% | 7,3% | 0.0645 | 0.1460 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.279221 | 55 | 52,7% | 6,8% | 0.0545 | 0.1360 |
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 80 | 51,2% | 5,3% | 0.0250 | 0.1065 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31524 | 67 | 44,8% | -1,1% | -0.1045 | -0.0230 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0383034 AND slope_strength low80@0.277751 | 52 | 50,0% | 11,5% | 0.2500 | 0.2885 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 59 | 45,8% | 7,3% | 0.1441 | 0.1825 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.27288 | 66 | 42,4% | 4,0% | 0.0606 | 0.0991 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.285581 | 79 | 41,8% | 3,3% | 0.0443 | 0.0828 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0876 | 77 | 44,2% | 10,8% | 0.3247 | 0.3247 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 71 | 43,7% | 10,3% | 0.3099 | 0.3099 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 57 | 40,4% | 7,0% | 0.2105 | 0.2105 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 78 | 35,9% | 2,6% | 0.0769 | 0.0769 |
| RR2_volz_high50 | 1 | vol_z high50@-0.319338 | 80 | 35,0% | 1,7% | 0.0500 | 0.0500 |

