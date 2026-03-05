# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 43 | 53,5% | 10,0% | 0.0698 | 0.1992 |
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 47 | 53,2% | 9,7% | 0.0638 | 0.1932 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271206 | 36 | 52,8% | 9,2% | 0.0556 | 0.1850 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24529 | 42 | 47,6% | 4,1% | -0.0476 | 0.0818 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.270919 | 35 | 48,6% | 13,2% | 0.2143 | 0.3301 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 42 | 42,9% | 7,5% | 0.0714 | 0.1873 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22997 | 42 | 42,9% | 7,5% | 0.0714 | 0.1873 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.280266 | 50 | 38,0% | 2,6% | -0.0500 | 0.0659 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9202 AND pullback_from_new_high_atr low50@1.05303 | 36 | 38,9% | 8,9% | 0.1667 | 0.2667 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 41 | 36,6% | 6,6% | 0.0976 | 0.1976 |
| RR2_rsi_high60 | 1 | rsi high60@59.9202 | 52 | 36,5% | 6,5% | 0.0962 | 0.1962 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 47 | 36,2% | 6,2% | 0.0851 | 0.1851 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 49 | 32,7% | 2,7% | -0.0204 | 0.0796 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 53 | 32,1% | 2,1% | -0.0377 | 0.0623 |

