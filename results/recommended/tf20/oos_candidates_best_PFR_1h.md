# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 43 | 51,2% | 8,1% | 0.0233 | 0.1628 |
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 51 | 51,0% | 8,0% | 0.0196 | 0.1591 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.270919 | 36 | 50,0% | 7,0% | 0.0000 | 0.1395 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23978 | 46 | 45,7% | 2,6% | -0.0870 | 0.0526 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.270416 | 35 | 45,7% | 10,8% | 0.1429 | 0.2694 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22747 | 44 | 40,9% | 6,0% | 0.0227 | 0.1492 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 42 | 40,5% | 5,5% | 0.0119 | 0.1384 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.286689 | 53 | 39,6% | 4,7% | -0.0094 | 0.1171 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.939 AND pullback_from_new_high_atr low50@1.05693 | 37 | 37,8% | 7,8% | 0.1351 | 0.2351 |
| RR2_rsi_high60 | 1 | rsi high60@59.939 | 52 | 36,5% | 6,5% | 0.0962 | 0.1962 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 48 | 35,4% | 5,4% | 0.0625 | 0.1625 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 41 | 34,1% | 4,1% | 0.0244 | 0.1244 |
| RR2_volz_high50 | 1 | vol_z high50@-0.29227 | 50 | 34,0% | 4,0% | 0.0200 | 0.1200 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 52 | 32,7% | 2,7% | -0.0192 | 0.0808 |

