# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.280999 | 56 | 60,7% | 11,5% | 0.2143 | 0.2299 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 63 | 60,3% | 11,1% | 0.2063 | 0.2220 |
| RR1_volz_high50 | 1 | vol_z high50@-0.290896 | 77 | 59,7% | 10,5% | 0.1948 | 0.2104 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.3187 | 62 | 50,0% | 0,8% | 0.0000 | 0.0156 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 61 | 50,8% | 11,0% | 0.2705 | 0.2746 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.292849 | 75 | 46,7% | 6,8% | 0.1667 | 0.1707 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.3122 | 60 | 46,7% | 6,8% | 0.1667 | 0.1707 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 58 | 43,1% | 9,5% | 0.2931 | 0.2847 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 67 | 41,8% | 8,2% | 0.2537 | 0.2453 |
| RR2_rsi_high60 | 1 | rsi high60@60.2994 | 65 | 40,0% | 6,4% | 0.2000 | 0.1916 |
| RR2_volz_high50 | 1 | vol_z high50@-0.333782 | 76 | 38,2% | 4,5% | 0.1447 | 0.1363 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 75 | 33,3% | -0,3% | -0.0000 | -0.0084 |

