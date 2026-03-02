# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0327714 | 47 | 53,2% | 8,5% | 0.0638 | 0.1697 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0327714 AND slope_strength low80@0.271314 | 40 | 52,5% | 7,8% | 0.0500 | 0.1559 |
| RR1_volz_high50 | 1 | vol_z high50@-0.237565 | 46 | 52,2% | 7,5% | 0.0435 | 0.1494 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24461 | 42 | 47,6% | 2,9% | -0.0476 | 0.0583 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0338931 AND slope_strength low80@0.271242 | 38 | 47,4% | 12,0% | 0.1842 | 0.3001 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23863 | 40 | 42,5% | 7,1% | 0.0625 | 0.1784 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 45 | 42,2% | 6,9% | 0.0556 | 0.1714 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 48 | 37,5% | 2,1% | -0.0625 | 0.0534 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0219 | 47 | 38,3% | 9,2% | 0.1489 | 0.2755 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04786 | 44 | 36,4% | 7,2% | 0.0909 | 0.2175 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0338931 | 43 | 34,9% | 5,8% | 0.0465 | 0.1731 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 50 | 32,0% | 2,9% | -0.0400 | 0.0866 |
| RR2_volz_high50 | 1 | vol_z high50@-0.288793 | 47 | 31,9% | 2,8% | -0.0426 | 0.0840 |

