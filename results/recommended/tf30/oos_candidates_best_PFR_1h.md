# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0315293 AND body_pct high60@0.179168 | 49 | 53,1% | 6,9% | 0.0612 | 0.1381 |
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0315293 AND clv high60@0.826752 | 47 | 51,1% | 4,9% | 0.0213 | 0.0982 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0315293 | 71 | 49,3% | 3,1% | -0.0141 | 0.0628 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0315293 AND slope_strength low80@0.274613 | 62 | 48,4% | 2,2% | -0.0323 | 0.0447 |
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 83 | 47,0% | 0,8% | -0.0602 | 0.0167 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31327 | 66 | 47,0% | 0,8% | -0.0606 | 0.0163 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.28322 | 64 | 45,3% | 5,5% | 0.1328 | 0.1364 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.272981 | 58 | 43,1% | 3,2% | 0.0776 | 0.0812 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 67 | 40,3% | 0,4% | 0.0075 | 0.0111 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.280266 | 86 | 38,4% | -1,5% | -0.0407 | -0.0371 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0377 | 83 | 42,2% | 8,3% | 0.2651 | 0.2500 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04786 | 74 | 41,9% | 8,1% | 0.2568 | 0.2417 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 83 | 38,6% | 4,7% | 0.1566 | 0.1416 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 65 | 35,4% | 1,6% | 0.0615 | 0.0465 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 83 | 31,3% | -2,5% | -0.0602 | -0.0753 |

