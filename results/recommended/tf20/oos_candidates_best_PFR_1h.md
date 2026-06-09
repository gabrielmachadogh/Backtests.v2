# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 53 | 47,2% | 4,5% | -0.0566 | 0.0895 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 42 | 45,2% | 2,5% | -0.0952 | 0.0508 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 46 | 43,5% | 0,8% | -0.1304 | 0.0156 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271349 | 38 | 42,1% | -0,6% | -0.1579 | -0.0118 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271314 | 37 | 40,5% | 5,7% | 0.0135 | 0.1414 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23978 | 45 | 40,0% | 5,1% | 0.0000 | 0.1279 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 41 | 36,6% | 1,7% | -0.0854 | 0.0425 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.277879 | 55 | 34,5% | -0,3% | -0.1364 | -0.0085 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9577 AND pullback_from_new_high_atr low50@1.05743 | 37 | 40,5% | 10,4% | 0.2162 | 0.3126 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 49 | 36,7% | 6,6% | 0.1020 | 0.1984 |
| RR2_rsi_high60 | 1 | rsi high60@59.9577 | 54 | 35,2% | 5,1% | 0.0556 | 0.1519 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 39 | 33,3% | 3,2% | -0.0000 | 0.0964 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 30,4% | 0,2% | -0.0893 | 0.0071 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 54 | 29,6% | -0,5% | -0.1111 | -0.0147 |

