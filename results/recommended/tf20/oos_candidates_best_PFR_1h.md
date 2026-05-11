# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 54 | 50,0% | 5,1% | 0.0000 | 0.1011 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 42 | 50,0% | 5,1% | 0.0000 | 0.1011 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.270919 | 37 | 45,9% | 1,0% | -0.0811 | 0.0200 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24461 | 49 | 44,9% | -0,0% | -0.1020 | -0.0009 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.270416 | 35 | 42,9% | 6,4% | 0.0714 | 0.1597 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23863 | 46 | 41,3% | 4,8% | 0.0326 | 0.1208 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 40 | 40,0% | 3,5% | 0.0000 | 0.0882 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.280266 | 53 | 37,7% | 1,3% | -0.0566 | 0.0316 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9009 AND pullback_from_new_high_atr low50@1.05303 | 38 | 39,5% | 6,9% | 0.1842 | 0.2083 |
| RR2_rsi_high60 | 1 | rsi high60@59.9009 | 56 | 37,5% | 5,0% | 0.1250 | 0.1491 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 49 | 36,7% | 4,2% | 0.1020 | 0.1261 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 39 | 35,9% | 3,4% | 0.0769 | 0.1010 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 52 | 34,6% | 2,1% | 0.0385 | 0.0626 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 57 | 31,6% | -1,0% | -0.0526 | -0.0285 |

