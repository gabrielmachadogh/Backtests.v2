# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 54,7% | 8,4% | 0.0938 | 0.1684 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.28011 | 57 | 54,4% | 8,1% | 0.0877 | 0.1623 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 81 | 51,9% | 5,6% | 0.0370 | 0.1117 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31327 | 68 | 45,6% | -0,7% | -0.0882 | -0.0136 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.278776 | 54 | 51,9% | 13,1% | 0.2963 | 0.3273 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 61 | 47,5% | 8,8% | 0.1885 | 0.2195 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24529 | 68 | 44,1% | 5,4% | 0.1029 | 0.1339 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.291583 | 81 | 43,2% | 4,5% | 0.0802 | 0.1113 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0791 | 76 | 42,1% | 9,3% | 0.2632 | 0.2792 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 69 | 42,0% | 9,2% | 0.2609 | 0.2769 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 58 | 41,4% | 8,6% | 0.2414 | 0.2574 |
| RR2_volz_high50 | 1 | vol_z high50@-0.319788 | 79 | 35,4% | 2,6% | 0.0633 | 0.0793 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 78 | 34,6% | 1,8% | 0.0385 | 0.0545 |

