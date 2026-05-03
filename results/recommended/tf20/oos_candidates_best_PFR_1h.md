# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.267522 | 53 | 49,1% | 4,7% | -0.0189 | 0.0948 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 41 | 48,8% | 4,5% | -0.0244 | 0.0892 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24461 | 49 | 44,9% | 0,6% | -0.1020 | 0.0116 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.270919 | 36 | 44,4% | 0,1% | -0.1111 | 0.0025 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.270667 | 35 | 42,9% | 6,4% | 0.0714 | 0.1597 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23498 | 47 | 42,6% | 6,1% | 0.0638 | 0.1521 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 40 | 40,0% | 3,5% | 0.0000 | 0.0882 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.283478 | 53 | 37,7% | 1,3% | -0.0566 | 0.0316 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9009 AND pullback_from_new_high_atr low50@1.05303 | 38 | 39,5% | 7,8% | 0.1842 | 0.2330 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 49 | 36,7% | 5,0% | 0.1020 | 0.1508 |
| RR2_rsi_high60 | 1 | rsi high60@59.9009 | 55 | 36,4% | 4,7% | 0.0909 | 0.1397 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 38 | 34,2% | 2,5% | 0.0263 | 0.0751 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 51 | 33,3% | 1,6% | -0.0000 | 0.0488 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 57 | 31,6% | -0,1% | -0.0526 | -0.0039 |

