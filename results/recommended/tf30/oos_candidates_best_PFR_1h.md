# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0361298 AND body_pct high60@0.179168 | 44 | 52,3% | 6,2% | 0.0455 | 0.1246 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 50,0% | 4,0% | 0.0000 | 0.0791 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.277751 | 56 | 50,0% | 4,0% | 0.0000 | 0.0791 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 85 | 49,4% | 3,4% | -0.0118 | 0.0674 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.3156 | 66 | 45,5% | -0,6% | -0.0909 | -0.0118 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.276009 | 53 | 47,2% | 8,4% | 0.1792 | 0.2091 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.28599 | 64 | 43,8% | 4,9% | 0.0938 | 0.1236 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 61 | 42,6% | 3,8% | 0.0656 | 0.0954 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.291583 | 85 | 41,2% | 2,4% | 0.0294 | 0.0593 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0535 | 79 | 44,3% | 10,2% | 0.3291 | 0.3059 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 73 | 43,8% | 9,7% | 0.3151 | 0.2918 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 59 | 39,0% | 4,9% | 0.1695 | 0.1462 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 81 | 37,0% | 2,9% | 0.1111 | 0.0879 |
| RR2_volz_high50 | 1 | vol_z high50@-0.319338 | 82 | 34,1% | 0,0% | 0.0244 | 0.0011 |

