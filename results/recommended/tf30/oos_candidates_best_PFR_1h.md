# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0350147 AND clv high60@0.826255 | 46 | 52,2% | 5,0% | 0.0435 | 0.0998 |
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0350147 AND body_pct high60@0.178797 | 48 | 52,1% | 4,9% | 0.0417 | 0.0980 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 68 | 50,0% | 2,8% | 0.0000 | 0.0563 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.27659 | 59 | 49,2% | 2,0% | -0.0169 | 0.0394 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31273 | 66 | 48,5% | 1,3% | -0.0303 | 0.0260 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 88 | 47,7% | 0,5% | -0.0455 | 0.0109 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.25839 | 66 | 48,5% | 7,6% | 0.2121 | 0.1902 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.274613 | 56 | 44,6% | 3,8% | 0.1161 | 0.0942 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 65 | 41,5% | 0,7% | 0.0385 | 0.0166 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.290896 | 88 | 40,9% | 0,0% | 0.0227 | 0.0008 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 75 | 44,0% | 9,2% | 0.3200 | 0.2745 |
| RR2_rsi_high60 | 1 | rsi high60@60.0456 | 82 | 43,9% | 9,1% | 0.3171 | 0.2716 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 82 | 39,0% | 4,2% | 0.1707 | 0.1253 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 64 | 37,5% | 2,7% | 0.1250 | 0.0795 |
| RR2_volz_high50 | 1 | vol_z high50@-0.292849 | 83 | 33,7% | -1,1% | 0.0120 | -0.0334 |

