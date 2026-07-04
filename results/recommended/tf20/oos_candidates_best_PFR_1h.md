# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 54 | 48,1% | 3,7% | -0.0370 | 0.0741 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 42 | 47,6% | 3,2% | -0.0476 | 0.0635 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271278 | 38 | 44,7% | 0,3% | -0.1053 | 0.0058 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.27011 | 44 | 43,2% | -1,3% | -0.1364 | -0.0253 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.271242 | 37 | 43,2% | 6,5% | 0.0811 | 0.1615 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24392 | 44 | 40,9% | 4,1% | 0.0227 | 0.1032 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 41 | 39,0% | 2,2% | -0.0244 | 0.0561 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.277879 | 56 | 35,7% | -1,1% | -0.1071 | -0.0267 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.9009 AND pullback_from_new_high_atr low50@1.05792 | 39 | 43,6% | 11,4% | 0.3077 | 0.3434 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05792 | 51 | 39,2% | 7,1% | 0.1765 | 0.2122 |
| RR2_rsi_high60 | 1 | rsi high60@59.9009 | 57 | 36,8% | 4,7% | 0.1053 | 0.1410 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 39 | 35,9% | 3,8% | 0.0769 | 0.1126 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 32,1% | 0,0% | -0.0357 | 0.0000 |
| RR2_volz_high50 | 1 | vol_z high50@-0.29227 | 55 | 30,9% | -1,2% | -0.0727 | -0.0370 |

