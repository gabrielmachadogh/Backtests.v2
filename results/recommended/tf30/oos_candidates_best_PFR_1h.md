# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0338931 AND body_pct high60@0.179539 | 48 | 52,1% | 5,6% | 0.0417 | 0.1121 |
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0338931 AND clv high60@0.825125 | 47 | 51,1% | 4,6% | 0.0213 | 0.0917 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 70 | 48,6% | 2,1% | -0.0286 | 0.0419 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31381 | 65 | 47,7% | 1,2% | -0.0462 | 0.0243 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0338931 AND slope_strength low80@0.276009 | 61 | 47,5% | 1,1% | -0.0492 | 0.0212 |
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 87 | 47,1% | 0,6% | -0.0575 | 0.0130 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.27011 | 64 | 46,9% | 6,7% | 0.1719 | 0.1682 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.273797 | 57 | 43,9% | 3,7% | 0.0965 | 0.0928 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 66 | 40,9% | 0,8% | 0.0227 | 0.0191 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.285581 | 86 | 39,5% | -0,6% | -0.0116 | -0.0153 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 76 | 43,4% | 8,8% | 0.3026 | 0.2650 |
| RR2_rsi_high60 | 1 | rsi high60@60.0456 | 83 | 43,4% | 8,8% | 0.3012 | 0.2636 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 83 | 38,6% | 4,0% | 0.1566 | 0.1190 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 65 | 36,9% | 2,3% | 0.1077 | 0.0701 |
| RR2_volz_high50 | 1 | vol_z high50@-0.292849 | 84 | 33,3% | -1,3% | -0.0000 | -0.0376 |

