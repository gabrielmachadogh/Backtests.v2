# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 44 | 52,3% | 9,2% | 0.0455 | 0.1850 |
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 48 | 52,1% | 9,1% | 0.0417 | 0.1812 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271206 | 37 | 51,4% | 8,3% | 0.0270 | 0.1666 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24529 | 43 | 46,5% | 3,5% | -0.0698 | 0.0698 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.270667 | 36 | 47,2% | 11,9% | 0.1806 | 0.2964 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 43 | 41,9% | 6,5% | 0.0465 | 0.1624 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22274 | 43 | 41,9% | 6,5% | 0.0465 | 0.1624 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.283478 | 51 | 37,3% | 1,9% | -0.0686 | 0.0472 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8624 AND pullback_from_new_high_atr low50@1.05693 | 37 | 37,8% | 7,8% | 0.1351 | 0.2351 |
| RR2_rsi_high60 | 1 | rsi high60@59.8624 | 53 | 35,8% | 5,8% | 0.0755 | 0.1755 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 42 | 35,7% | 5,7% | 0.0714 | 0.1714 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 48 | 35,4% | 5,4% | 0.0625 | 0.1625 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 53 | 32,1% | 2,1% | -0.0377 | 0.0623 |
| RR2_volz_high50 | 1 | vol_z high50@-0.29227 | 50 | 32,0% | 2,0% | -0.0400 | 0.0600 |

