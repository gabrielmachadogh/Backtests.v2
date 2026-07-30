# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0372324 AND body_pct high60@0.178377 | 44 | 52,3% | 6,3% | 0.0455 | 0.1257 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 63 | 50,8% | 4,8% | 0.0159 | 0.0962 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 82 | 50,0% | 4,0% | 0.0000 | 0.0803 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.27717 | 56 | 50,0% | 4,0% | 0.0000 | 0.0803 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31434 | 67 | 46,3% | 0,3% | -0.0746 | 0.0057 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0383034 AND slope_strength low80@0.275429 | 53 | 47,2% | 8,5% | 0.1792 | 0.2133 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 67 | 44,8% | 6,1% | 0.1194 | 0.1535 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 60 | 43,3% | 4,7% | 0.0833 | 0.1174 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.291583 | 82 | 41,5% | 2,8% | 0.0366 | 0.0707 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0298 | 78 | 43,6% | 10,0% | 0.3077 | 0.2999 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05792 | 72 | 43,1% | 9,5% | 0.2917 | 0.2839 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 59 | 39,0% | 5,4% | 0.1695 | 0.1617 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 80 | 36,2% | 2,7% | 0.0875 | 0.0797 |
| RR2_volz_high50 | 1 | vol_z high50@-0.310795 | 79 | 34,2% | 0,6% | 0.0253 | 0.0175 |

