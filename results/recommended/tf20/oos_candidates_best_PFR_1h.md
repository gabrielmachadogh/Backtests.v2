# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.239186 | 56 | 48,2% | 3,1% | -0.0357 | 0.0611 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.271171 | 38 | 47,4% | 2,2% | -0.0526 | 0.0441 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24324 | 45 | 46,7% | 1,5% | -0.0667 | 0.0301 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 41 | 46,3% | 1,2% | -0.0732 | 0.0236 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22997 | 43 | 46,5% | 7,6% | 0.1628 | 0.1906 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.270919 | 37 | 43,2% | 4,4% | 0.0811 | 0.1089 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 40 | 40,0% | 1,1% | 0.0000 | 0.0278 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.275492 | 59 | 39,0% | 0,1% | -0.0254 | 0.0024 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0535 AND pullback_from_new_high_atr low50@1.04572 | 37 | 51,4% | 16,9% | 0.5405 | 0.5061 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04572 | 49 | 44,9% | 10,4% | 0.3469 | 0.3125 |
| RR2_rsi_high60 | 1 | rsi high60@60.0535 | 53 | 43,4% | 8,9% | 0.3019 | 0.2674 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 38 | 36,8% | 2,4% | 0.1053 | 0.0708 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 35,7% | 1,2% | 0.0714 | 0.0369 |
| RR2_volz_high50 | 1 | vol_z high50@-0.288793 | 57 | 33,3% | -1,1% | -0.0000 | -0.0345 |

