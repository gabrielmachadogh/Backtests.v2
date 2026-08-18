# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0371925 AND body_pct high60@0.178426 | 44 | 52,3% | 5,9% | 0.0455 | 0.1179 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 63 | 50,8% | 4,4% | 0.0159 | 0.0883 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 84 | 50,0% | 3,6% | 0.0000 | 0.0725 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.275429 | 56 | 50,0% | 3,6% | 0.0000 | 0.0725 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31488 | 65 | 46,2% | -0,2% | -0.0769 | -0.0045 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.272981 | 53 | 47,2% | 8,1% | 0.1792 | 0.2018 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.25977 | 65 | 44,6% | 5,5% | 0.1154 | 0.1379 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 60 | 43,3% | 4,2% | 0.0833 | 0.1059 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.290896 | 84 | 41,7% | 2,6% | 0.0417 | 0.0642 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.062 | 78 | 43,6% | 10,3% | 0.3077 | 0.3077 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 72 | 43,1% | 9,7% | 0.2917 | 0.2917 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 59 | 39,0% | 5,6% | 0.1695 | 0.1695 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 81 | 35,8% | 2,5% | 0.0741 | 0.0741 |
| RR2_volz_high50 | 1 | vol_z high50@-0.310795 | 81 | 34,6% | 1,2% | 0.0370 | 0.0370 |

