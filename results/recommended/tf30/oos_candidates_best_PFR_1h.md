# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0327714 AND body_pct high60@0.17991 | 49 | 53,1% | 6,5% | 0.0612 | 0.1307 |
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0327714 AND clv high60@0.825441 | 48 | 52,1% | 5,6% | 0.0417 | 0.1111 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0327714 | 72 | 50,0% | 3,5% | 0.0000 | 0.0694 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0327714 AND slope_strength low80@0.273797 | 63 | 49,2% | 2,7% | -0.0159 | 0.0536 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.30495 | 67 | 47,8% | 1,2% | -0.0448 | 0.0247 |
| RR1_volz_high50 | 1 | vol_z high50@-0.239186 | 82 | 46,3% | -0,2% | -0.0732 | -0.0037 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 65 | 46,2% | 6,3% | 0.1538 | 0.1575 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.271349 | 59 | 44,1% | 4,2% | 0.1017 | 0.1053 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 68 | 41,2% | 1,3% | 0.0294 | 0.0330 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.275492 | 85 | 37,6% | -2,2% | -0.0588 | -0.0552 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0535 AND pullback_from_new_high_atr low50@1.04658 | 59 | 45,8% | 11,4% | 0.3729 | 0.3430 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04658 | 75 | 42,7% | 8,3% | 0.2800 | 0.2501 |
| RR2_rsi_high60 | 1 | rsi high60@60.0535 | 83 | 42,2% | 7,8% | 0.2651 | 0.2352 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 84 | 39,3% | 5,0% | 0.1786 | 0.1487 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 66 | 36,4% | 2,0% | 0.0909 | 0.0611 |
| RR2_volz_high50 | 1 | vol_z high50@-0.290896 | 83 | 31,3% | -3,0% | -0.0602 | -0.0901 |

