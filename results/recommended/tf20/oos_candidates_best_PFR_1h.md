# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 43 | 51,2% | 8,1% | 0.0233 | 0.1628 |
| RR1_volz_high50 | 1 | vol_z high50@-0.259552 | 51 | 51,0% | 8,0% | 0.0196 | 0.1591 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.271171 | 36 | 50,0% | 7,0% | 0.0000 | 0.1395 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24324 | 45 | 46,7% | 3,6% | -0.0667 | 0.0729 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.270667 | 36 | 47,2% | 11,1% | 0.1806 | 0.2769 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22274 | 44 | 43,2% | 7,0% | 0.0795 | 0.1759 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 43 | 41,9% | 5,7% | 0.0465 | 0.1429 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.283478 | 52 | 38,5% | 2,3% | -0.0385 | 0.0579 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9009 AND pullback_from_new_high_atr low50@1.05303 | 37 | 37,8% | 7,8% | 0.1351 | 0.2351 |
| RR2_rsi_high60 | 1 | rsi high60@59.9009 | 53 | 35,8% | 5,8% | 0.0755 | 0.1755 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 48 | 35,4% | 5,4% | 0.0625 | 0.1625 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 41 | 34,1% | 4,1% | 0.0244 | 0.1244 |
| RR2_volz_high50 | 1 | vol_z high50@-0.292849 | 51 | 33,3% | 3,3% | -0.0000 | 0.1000 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 53 | 32,1% | 2,1% | -0.0377 | 0.0623 |

