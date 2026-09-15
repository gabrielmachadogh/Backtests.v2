# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0361298 AND body_pct high60@0.178229 | 46 | 54,3% | 6,5% | 0.0870 | 0.1298 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.27659 | 57 | 52,6% | 4,8% | 0.0526 | 0.0955 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 65 | 52,3% | 4,5% | 0.0462 | 0.0890 |
| RR1_volz_high50 | 1 | vol_z high50@-0.285581 | 87 | 50,6% | 2,7% | 0.0115 | 0.0544 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31768 | 64 | 46,9% | -1,0% | -0.0625 | -0.0196 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.274613 | 54 | 48,1% | 8,1% | 0.2037 | 0.2037 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.31327 | 61 | 44,3% | 4,3% | 0.1066 | 0.1066 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 62 | 43,5% | 3,5% | 0.0887 | 0.0887 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.292849 | 87 | 41,4% | 1,4% | 0.0345 | 0.0345 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 74 | 44,6% | 9,2% | 0.3378 | 0.2763 |
| RR2_rsi_high60 | 1 | rsi high60@60.0456 | 79 | 44,3% | 8,9% | 0.3291 | 0.2676 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 81 | 39,5% | 4,1% | 0.1852 | 0.1236 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 61 | 39,3% | 4,0% | 0.1803 | 0.1188 |
| RR2_volz_high50 | 1 | vol_z high50@-0.310795 | 82 | 35,4% | -0,0% | 0.0610 | -0.0006 |

