# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.280888 | 57 | 59,6% | 10,8% | 0.1930 | 0.2162 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 59,4% | 10,5% | 0.1875 | 0.2108 |
| RR1_volz_high50 | 1 | vol_z high50@-0.285581 | 76 | 59,2% | 10,4% | 0.1842 | 0.2075 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.32072 | 62 | 48,4% | -0,5% | -0.0323 | -0.0090 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 61 | 50,8% | 10,5% | 0.2705 | 0.2624 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.293428 | 77 | 46,8% | 6,4% | 0.1688 | 0.1608 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.31601 | 59 | 45,8% | 5,4% | 0.1441 | 0.1360 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 58 | 43,1% | 8,9% | 0.2931 | 0.2681 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 69 | 42,0% | 7,9% | 0.2609 | 0.2359 |
| RR2_rsi_high60 | 1 | rsi high60@60.219 | 67 | 40,3% | 6,1% | 0.2090 | 0.1840 |
| RR2_volz_high50 | 1 | vol_z high50@-0.338748 | 78 | 38,5% | 4,3% | 0.1538 | 0.1288 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 76 | 34,2% | 0,0% | 0.0263 | 0.0013 |

