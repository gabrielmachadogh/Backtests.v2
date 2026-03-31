# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 51 | 51,0% | 7,3% | 0.0196 | 0.1460 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 43 | 48,8% | 5,2% | -0.0233 | 0.1032 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.270416 | 36 | 47,2% | 3,5% | -0.0556 | 0.0709 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24392 | 45 | 44,4% | 0,8% | -0.1111 | 0.0153 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22497 | 43 | 41,9% | 5,7% | 0.0465 | 0.1429 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 41 | 41,5% | 5,3% | 0.0366 | 0.1330 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.286689 | 52 | 40,4% | 4,2% | 0.0096 | 0.1060 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8816 AND pullback_from_new_high_atr low50@1.05743 | 38 | 36,8% | 6,0% | 0.1053 | 0.1793 |
| RR2_rsi_high60 | 1 | rsi high60@59.8816 | 53 | 35,8% | 5,0% | 0.0755 | 0.1495 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 40 | 35,0% | 4,1% | 0.0500 | 0.1241 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 49 | 34,7% | 3,8% | 0.0408 | 0.1149 |
| RR2_volz_high50 | 1 | vol_z high50@-0.292849 | 50 | 34,0% | 3,1% | 0.0200 | 0.0941 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 53 | 32,1% | 1,2% | -0.0377 | 0.0363 |

