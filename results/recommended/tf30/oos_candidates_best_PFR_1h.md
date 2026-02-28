# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0383034 AND slope_strength low80@0.282347 | 57 | 61,4% | 11,8% | 0.2281 | 0.2359 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 62 | 61,3% | 11,7% | 0.2258 | 0.2337 |
| RR1_volz_high50 | 1 | vol_z high50@-0.285581 | 73 | 60,3% | 10,7% | 0.2055 | 0.2134 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.32407 | 60 | 50,0% | 0,4% | 0.0000 | 0.0079 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0393346 | 60 | 51,7% | 11,5% | 0.2917 | 0.2876 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.29909 | 60 | 48,3% | 8,2% | 0.2083 | 0.2042 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.293428 | 74 | 47,3% | 7,1% | 0.1824 | 0.1783 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 58 | 43,1% | 9,2% | 0.2931 | 0.2762 |
| RR2_rsi_high60 | 1 | rsi high60@60.1445 | 67 | 41,8% | 7,9% | 0.2537 | 0.2368 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05852 | 67 | 41,8% | 7,9% | 0.2537 | 0.2368 |
| RR2_volz_high50 | 1 | vol_z high50@-0.338748 | 75 | 38,7% | 4,8% | 0.1600 | 0.1431 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 74 | 33,8% | -0,1% | 0.0135 | -0.0034 |

