# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.2339 | 46 | 45,7% | 1,4% | -0.0870 | 0.0288 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 43 | 44,2% | -0,0% | -0.1163 | -0.0005 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.268038 | 39 | 43,6% | -0,6% | -0.1282 | -0.0124 |
| RR1_volz_high50 | 1 | vol_z high50@-0.234643 | 57 | 42,1% | -2,1% | -0.1579 | -0.0421 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22051 | 44 | 47,7% | 7,5% | 0.1932 | 0.1877 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.268038 | 39 | 38,5% | -1,8% | -0.0385 | -0.0439 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 43 | 37,2% | -3,0% | -0.0698 | -0.0752 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.259552 | 60 | 36,7% | -3,6% | -0.0833 | -0.0888 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9944 AND pullback_from_new_high_atr low50@1.04572 | 42 | 45,2% | 10,4% | 0.3571 | 0.3122 |
| RR2_rsi_high60 | 1 | rsi high60@59.9944 | 58 | 41,4% | 6,5% | 0.2414 | 0.1964 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04572 | 51 | 41,2% | 6,3% | 0.2353 | 0.1904 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 53 | 37,7% | 2,9% | 0.1321 | 0.0871 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 41 | 34,1% | -0,7% | 0.0244 | -0.0206 |
| RR2_volz_high50 | 1 | vol_z high50@-0.283007 | 58 | 31,0% | -3,8% | -0.0690 | -0.1139 |

