# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 52 | 50,0% | 7,0% | 0.0000 | 0.1395 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 42 | 50,0% | 7,0% | 0.0000 | 0.1395 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.270667 | 35 | 48,6% | 5,5% | -0.0286 | 0.1110 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24209 | 45 | 44,4% | 1,4% | -0.1111 | 0.0284 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.270164 | 35 | 45,7% | 9,6% | 0.1429 | 0.2392 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23248 | 43 | 41,9% | 5,7% | 0.0465 | 0.1429 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 42 | 40,5% | 4,3% | 0.0119 | 0.1083 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.283478 | 51 | 39,2% | 3,1% | -0.0196 | 0.0768 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8816 AND pullback_from_new_high_atr low50@1.05743 | 37 | 37,8% | 6,6% | 0.1351 | 0.1976 |
| RR2_rsi_high60 | 1 | rsi high60@59.8816 | 52 | 36,5% | 5,3% | 0.0962 | 0.1587 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 48 | 35,4% | 4,2% | 0.0625 | 0.1250 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 40 | 35,0% | 3,8% | 0.0500 | 0.1125 |
| RR2_volz_high50 | 1 | vol_z high50@-0.292849 | 50 | 34,0% | 2,8% | 0.0200 | 0.0825 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 52 | 32,7% | 1,4% | -0.0192 | 0.0433 |

