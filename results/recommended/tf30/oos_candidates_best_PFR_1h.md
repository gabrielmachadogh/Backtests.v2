# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0361298 AND body_pct high60@0.178229 | 45 | 53,3% | 5,9% | 0.0667 | 0.1170 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.27659 | 56 | 51,8% | 4,3% | 0.0357 | 0.0861 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 51,6% | 4,1% | 0.0312 | 0.0816 |
| RR1_volz_high50 | 1 | vol_z high50@-0.285581 | 86 | 50,0% | 2,5% | 0.0000 | 0.0504 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31768 | 64 | 46,9% | -0,6% | -0.0625 | -0.0121 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.274613 | 53 | 49,1% | 8,8% | 0.2264 | 0.2190 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 61 | 44,3% | 4,0% | 0.1066 | 0.0991 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.31327 | 61 | 44,3% | 4,0% | 0.1066 | 0.0991 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.292849 | 86 | 41,9% | 1,6% | 0.0465 | 0.0390 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 73 | 45,2% | 9,8% | 0.3562 | 0.2946 |
| RR2_rsi_high60 | 1 | rsi high60@60.0298 | 79 | 44,3% | 8,9% | 0.3291 | 0.2676 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 60 | 40,0% | 4,6% | 0.2000 | 0.1385 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 82 | 39,0% | 3,6% | 0.1707 | 0.1092 |
| RR2_volz_high50 | 1 | vol_z high50@-0.315066 | 82 | 35,4% | -0,0% | 0.0610 | -0.0006 |

