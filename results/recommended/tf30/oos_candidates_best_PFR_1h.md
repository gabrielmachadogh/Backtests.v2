# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0350409 AND body_pct high60@0.178426 | 48 | 54,2% | 6,3% | 0.0833 | 0.1262 |
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0350409 AND clv high60@0.828244 | 45 | 53,3% | 5,5% | 0.0667 | 0.1095 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 67 | 52,2% | 4,4% | 0.0448 | 0.0876 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.278331 | 58 | 51,7% | 3,9% | 0.0345 | 0.0773 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 87 | 50,6% | 2,7% | 0.0115 | 0.0544 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31667 | 65 | 47,7% | -0,2% | -0.0462 | -0.0033 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.27659 | 55 | 47,3% | 6,5% | 0.1818 | 0.1633 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.31273 | 62 | 45,2% | 4,4% | 0.1290 | 0.1105 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 43,8% | 3,0% | 0.0938 | 0.0752 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.291583 | 87 | 42,5% | 1,8% | 0.0632 | 0.0447 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 75 | 45,3% | 9,5% | 0.3600 | 0.2837 |
| RR2_rsi_high60 | 1 | rsi high60@60.0219 | 81 | 44,4% | 8,6% | 0.3333 | 0.2570 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 63 | 39,7% | 3,8% | 0.1905 | 0.1141 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 81 | 39,5% | 3,6% | 0.1852 | 0.1088 |
| RR2_volz_high50 | 1 | vol_z high50@-0.315066 | 84 | 35,7% | -0,2% | 0.0714 | -0.0049 |

