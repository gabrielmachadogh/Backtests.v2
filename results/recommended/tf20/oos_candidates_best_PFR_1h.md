# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23211 | 46 | 45,7% | 1,4% | -0.0870 | 0.0288 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 43 | 44,2% | -0,0% | -0.1163 | -0.0005 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.269444 | 39 | 43,6% | -0,6% | -0.1282 | -0.0124 |
| RR1_volz_high50 | 1 | vol_z high50@-0.237565 | 58 | 43,1% | -1,1% | -0.1379 | -0.0221 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.21605 | 44 | 47,7% | 8,2% | 0.1932 | 0.2042 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.268507 | 38 | 39,5% | -0,1% | -0.0132 | -0.0022 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 42 | 38,1% | -1,5% | -0.0476 | -0.0366 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 60 | 36,7% | -2,9% | -0.0833 | -0.0723 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9993 AND pullback_from_new_high_atr low50@1.04572 | 41 | 46,3% | 12,6% | 0.3902 | 0.3790 |
| RR2_rsi_high60 | 1 | rsi high60@59.9993 | 58 | 41,4% | 7,7% | 0.2414 | 0.2301 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04572 | 51 | 41,2% | 7,5% | 0.2353 | 0.2241 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 54 | 35,2% | 1,5% | 0.0556 | 0.0443 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 41 | 34,1% | 0,4% | 0.0244 | 0.0132 |
| RR2_volz_high50 | 1 | vol_z high50@-0.286218 | 59 | 30,5% | -3,2% | -0.0847 | -0.0960 |

