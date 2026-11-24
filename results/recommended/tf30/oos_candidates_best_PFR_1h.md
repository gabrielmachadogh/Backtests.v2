# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0338931 AND body_pct high60@0.180502 | 48 | 52,1% | 6,3% | 0.0417 | 0.1250 |
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0338931 AND clv high60@0.826255 | 47 | 51,1% | 5,2% | 0.0213 | 0.1046 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 71 | 49,3% | 3,5% | -0.0141 | 0.0692 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0338931 AND slope_strength low80@0.272981 | 62 | 48,4% | 2,6% | -0.0323 | 0.0511 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.28322 | 69 | 47,8% | 2,0% | -0.0435 | 0.0399 |
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 82 | 45,1% | -0,7% | -0.0976 | -0.0142 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24461 | 67 | 44,8% | 5,6% | 0.1194 | 0.1411 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271314 | 58 | 43,1% | 4,0% | 0.0776 | 0.0993 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 67 | 40,3% | 1,2% | 0.0075 | 0.0292 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 84 | 36,9% | -2,2% | -0.0774 | -0.0556 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0298 AND pullback_from_new_high_atr low50@1.04786 | 60 | 45,0% | 10,7% | 0.3500 | 0.3201 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04786 | 76 | 42,1% | 7,8% | 0.2632 | 0.2333 |
| RR2_rsi_high60 | 1 | rsi high60@60.0298 | 84 | 41,7% | 7,3% | 0.2500 | 0.2201 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 85 | 38,8% | 4,5% | 0.1647 | 0.1349 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 66 | 36,4% | 2,0% | 0.0909 | 0.0611 |
| RR2_volz_high50 | 1 | vol_z high50@-0.288793 | 83 | 31,3% | -3,0% | -0.0602 | -0.0901 |

