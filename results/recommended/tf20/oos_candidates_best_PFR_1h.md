# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 52 | 50,0% | 6,2% | 0.0000 | 0.1236 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 42 | 47,6% | 3,8% | -0.0476 | 0.0760 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.25253 | 46 | 45,7% | 1,8% | -0.0870 | 0.0366 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271242 | 37 | 43,2% | -0,6% | -0.1351 | -0.0115 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24094 | 45 | 42,2% | 6,2% | 0.0556 | 0.1544 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271206 | 36 | 41,7% | 5,6% | 0.0417 | 0.1405 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 41 | 39,0% | 3,0% | -0.0244 | 0.0744 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.277879 | 54 | 37,0% | 1,0% | -0.0741 | 0.0248 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8816 AND pullback_from_new_high_atr low50@1.05743 | 38 | 39,5% | 8,1% | 0.1842 | 0.2445 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 50 | 36,0% | 4,7% | 0.0800 | 0.1402 |
| RR2_rsi_high60 | 1 | rsi high60@59.8816 | 56 | 35,7% | 4,4% | 0.0714 | 0.1317 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 39 | 33,3% | 2,0% | -0.0000 | 0.0602 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 53 | 32,1% | 0,8% | -0.0377 | 0.0225 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 58 | 31,0% | -0,3% | -0.0690 | -0.0087 |

