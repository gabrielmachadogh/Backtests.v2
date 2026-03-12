# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.280888 | 57 | 59,6% | 11,2% | 0.1930 | 0.2242 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 59,4% | 10,9% | 0.1875 | 0.2188 |
| RR1_volz_high50 | 1 | vol_z high50@-0.285581 | 75 | 58,7% | 10,2% | 0.1733 | 0.2046 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.32072 | 62 | 48,4% | -0,1% | -0.0323 | -0.0010 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 61 | 50,8% | 11,0% | 0.2705 | 0.2746 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.293428 | 76 | 46,1% | 6,2% | 0.1513 | 0.1554 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.31601 | 59 | 45,8% | 5,9% | 0.1441 | 0.1481 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 59 | 42,4% | 9,0% | 0.2712 | 0.2712 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 68 | 41,2% | 7,8% | 0.2353 | 0.2353 |
| RR2_rsi_high60 | 1 | rsi high60@60.2994 | 66 | 39,4% | 6,1% | 0.1818 | 0.1818 |
| RR2_volz_high50 | 1 | vol_z high50@-0.333782 | 77 | 37,7% | 4,3% | 0.1299 | 0.1299 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 75 | 33,3% | 0,0% | -0.0000 | 0.0000 |

