# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24094 | 46 | 45,7% | 1,0% | -0.0870 | 0.0194 |
| RR1_volz_high50 | 1 | vol_z high50@-0.237565 | 57 | 45,6% | 0,9% | -0.0877 | 0.0187 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 42 | 45,2% | 0,6% | -0.0952 | 0.0111 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.270164 | 38 | 44,7% | 0,1% | -0.1053 | 0.0011 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22497 | 44 | 45,5% | 7,0% | 0.1364 | 0.1748 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.269913 | 37 | 40,5% | 2,1% | 0.0135 | 0.0520 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 41 | 39,0% | 0,6% | -0.0244 | 0.0141 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.259552 | 59 | 37,3% | -1,2% | -0.0678 | -0.0293 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0042 AND pullback_from_new_high_atr low50@1.04572 | 39 | 48,7% | 14,6% | 0.4615 | 0.4388 |
| RR2_rsi_high60 | 1 | rsi high60@60.0042 | 57 | 42,1% | 8,0% | 0.2632 | 0.2404 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04572 | 50 | 42,0% | 7,9% | 0.2600 | 0.2373 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 54 | 35,2% | 1,1% | 0.0556 | 0.0328 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 40 | 35,0% | 0,9% | 0.0500 | 0.0273 |
| RR2_volz_high50 | 1 | vol_z high50@-0.283007 | 57 | 31,6% | -2,5% | -0.0526 | -0.0754 |

