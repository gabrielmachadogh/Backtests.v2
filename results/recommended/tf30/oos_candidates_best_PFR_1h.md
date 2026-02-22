# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0383034 AND slope_strength low80@0.282347 | 57 | 61,4% | 11,4% | 0.2281 | 0.2281 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 62 | 61,3% | 11,3% | 0.2258 | 0.2258 |
| RR1_volz_high50 | 1 | vol_z high50@-0.285581 | 73 | 60,3% | 10,3% | 0.2055 | 0.2055 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.32407 | 59 | 50,8% | 0,8% | 0.0169 | 0.0169 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.039568 | 59 | 52,5% | 11,6% | 0.3136 | 0.2890 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.31601 | 57 | 47,4% | 6,4% | 0.1842 | 0.1596 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.292849 | 74 | 47,3% | 6,3% | 0.1824 | 0.1578 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0393346 | 58 | 43,1% | 9,2% | 0.2931 | 0.2762 |
| RR2_rsi_high60 | 1 | rsi high60@60.1149 | 68 | 41,2% | 7,3% | 0.2353 | 0.2183 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05912 | 68 | 41,2% | 7,3% | 0.2353 | 0.2183 |
| RR2_volz_high50 | 1 | vol_z high50@-0.333782 | 75 | 38,7% | 4,8% | 0.1600 | 0.1431 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 75 | 33,3% | -0,6% | -0.0000 | -0.0169 |

