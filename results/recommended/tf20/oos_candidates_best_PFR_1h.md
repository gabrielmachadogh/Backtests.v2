# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 42 | 47,6% | 3,8% | -0.0476 | 0.0760 |
| RR1_volz_high50 | 1 | vol_z high50@-0.259552 | 55 | 47,3% | 3,5% | -0.0545 | 0.0691 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24529 | 46 | 45,7% | 1,8% | -0.0870 | 0.0366 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.271206 | 37 | 43,2% | -0,6% | -0.1351 | -0.0115 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23748 | 45 | 42,2% | 6,2% | 0.0556 | 0.1544 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.271171 | 36 | 41,7% | 5,6% | 0.0417 | 0.1405 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 41 | 39,0% | 3,0% | -0.0244 | 0.0744 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.280266 | 55 | 36,4% | 0,3% | -0.0909 | 0.0079 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9202 AND pullback_from_new_high_atr low50@1.05792 | 38 | 39,5% | 9,4% | 0.1842 | 0.2806 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05792 | 50 | 36,0% | 5,9% | 0.0800 | 0.1764 |
| RR2_rsi_high60 | 1 | rsi high60@59.9202 | 55 | 34,5% | 4,4% | 0.0364 | 0.1327 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 39 | 33,3% | 3,2% | -0.0000 | 0.0964 |
| RR2_volz_high50 | 1 | vol_z high50@-0.290896 | 53 | 30,2% | 0,1% | -0.0943 | 0.0020 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 57 | 29,8% | -0,3% | -0.1053 | -0.0089 |

