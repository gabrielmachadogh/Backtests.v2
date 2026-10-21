# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0327714 AND body_pct high60@0.180282 | 48 | 52,1% | 5,6% | 0.0417 | 0.1121 |
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0327714 AND clv high60@0.825758 | 47 | 51,1% | 4,6% | 0.0213 | 0.0917 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0327714 | 70 | 48,6% | 2,1% | -0.0286 | 0.0419 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0327714 AND slope_strength low80@0.275429 | 61 | 47,5% | 1,1% | -0.0492 | 0.0212 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31488 | 64 | 46,9% | 0,4% | -0.0625 | 0.0079 |
| RR1_volz_high50 | 1 | vol_z high50@-0.259552 | 86 | 46,5% | 0,0% | -0.0698 | 0.0007 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.28322 | 63 | 46,0% | 5,9% | 0.1508 | 0.1471 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.272981 | 57 | 43,9% | 3,7% | 0.0965 | 0.0928 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 66 | 40,9% | 0,8% | 0.0227 | 0.0191 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.280266 | 85 | 38,8% | -1,3% | -0.0294 | -0.0331 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0219 | 83 | 43,4% | 8,8% | 0.3012 | 0.2636 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04913 | 74 | 43,2% | 8,7% | 0.2973 | 0.2597 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 83 | 38,6% | 4,0% | 0.1566 | 0.1190 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0327714 | 65 | 36,9% | 2,3% | 0.1077 | 0.0701 |
| RR2_volz_high50 | 1 | vol_z high50@-0.29227 | 83 | 32,5% | -2,1% | -0.0241 | -0.0617 |

