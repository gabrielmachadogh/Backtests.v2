# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 46 | 52,2% | 8,6% | 0.0435 | 0.1729 |
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 46 | 52,2% | 8,6% | 0.0435 | 0.1729 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0338931 AND slope_strength low80@0.271278 | 39 | 51,3% | 7,8% | 0.0256 | 0.1551 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24209 | 43 | 46,5% | 3,0% | -0.0698 | 0.0596 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.271206 | 37 | 45,9% | 11,8% | 0.1486 | 0.2950 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23248 | 41 | 41,5% | 7,3% | 0.0366 | 0.1829 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 44 | 40,9% | 6,8% | 0.0227 | 0.1691 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.275492 | 49 | 36,7% | 2,6% | -0.0816 | 0.0647 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@59.9765 | 49 | 36,7% | 7,6% | 0.1020 | 0.2286 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04913 | 44 | 36,4% | 7,2% | 0.0909 | 0.2175 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 42 | 35,7% | 6,6% | 0.0714 | 0.1980 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 51 | 31,4% | 2,3% | -0.0588 | 0.0678 |
| RR2_volz_high50 | 1 | vol_z high50@-0.290896 | 48 | 31,2% | 2,1% | -0.0625 | 0.0641 |

