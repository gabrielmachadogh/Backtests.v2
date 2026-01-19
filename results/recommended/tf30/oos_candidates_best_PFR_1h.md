# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0393346 AND slope_strength low80@0.282965 | 56 | 60,7% | 10,7% | 0.2143 | 0.2143 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0393346 | 61 | 60,7% | 10,7% | 0.2131 | 0.2131 |
| RR1_volz_high50 | 1 | vol_z high50@-0.290896 | 75 | 60,0% | 10,0% | 0.2000 | 0.2000 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.32072 | 60 | 51,7% | 1,7% | 0.0333 | 0.0333 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.039568 | 58 | 51,7% | 11,2% | 0.2931 | 0.2807 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.31601 | 57 | 47,4% | 6,9% | 0.1842 | 0.1718 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.292849 | 73 | 46,6% | 6,1% | 0.1644 | 0.1520 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0393346 | 57 | 43,9% | 9,7% | 0.3158 | 0.2901 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05912 | 67 | 41,8% | 7,6% | 0.2537 | 0.2281 |
| RR2_rsi_high60 | 1 | rsi high60@60.1149 | 68 | 41,2% | 7,0% | 0.2353 | 0.2097 |
| RR2_volz_high50 | 1 | vol_z high50@-0.333782 | 74 | 39,2% | 5,0% | 0.1757 | 0.1500 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 74 | 33,8% | -0,4% | 0.0135 | -0.0121 |

