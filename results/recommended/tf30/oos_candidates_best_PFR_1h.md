# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.280555 | 55 | 58,2% | 9,7% | 0.1636 | 0.1939 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 62 | 58,1% | 9,6% | 0.1613 | 0.1916 |
| RR1_volz_high50 | 1 | vol_z high50@-0.285581 | 77 | 55,8% | 7,4% | 0.1169 | 0.1472 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31488 | 68 | 47,1% | -1,4% | -0.0588 | -0.0285 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.279221 | 52 | 53,8% | 13,7% | 0.3462 | 0.3422 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 59 | 49,2% | 9,0% | 0.2288 | 0.2249 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.25977 | 68 | 45,6% | 5,4% | 0.1397 | 0.1358 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.292849 | 77 | 45,5% | 5,3% | 0.1364 | 0.1324 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 56 | 42,9% | 8,7% | 0.2857 | 0.2613 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 69 | 42,0% | 7,9% | 0.2609 | 0.2365 |
| RR2_rsi_high60 | 1 | rsi high60@60.0791 | 74 | 41,9% | 7,7% | 0.2568 | 0.2324 |
| RR2_volz_high50 | 1 | vol_z high50@-0.320238 | 75 | 37,3% | 3,2% | 0.1200 | 0.0956 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 79 | 35,4% | 1,3% | 0.0633 | 0.0389 |

