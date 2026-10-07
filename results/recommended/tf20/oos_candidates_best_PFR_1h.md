# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 57 | 47,4% | 1,6% | -0.0526 | 0.0325 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23863 | 47 | 46,8% | 1,1% | -0.0638 | 0.0213 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 42 | 45,2% | -0,5% | -0.0952 | -0.0101 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.270416 | 38 | 44,7% | -1,0% | -0.1053 | -0.0202 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22051 | 46 | 47,8% | 8,3% | 0.1957 | 0.2066 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.270164 | 37 | 40,5% | 1,0% | 0.0135 | 0.0245 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 41 | 39,0% | -0,5% | -0.0244 | -0.0134 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 59 | 39,0% | -0,6% | -0.0254 | -0.0144 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0219 AND pullback_from_new_high_atr low50@1.04486 | 38 | 50,0% | 15,9% | 0.5000 | 0.4773 |
| RR2_rsi_high60 | 1 | rsi high60@60.0219 | 56 | 42,9% | 8,8% | 0.2857 | 0.2630 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04486 | 49 | 42,9% | 8,8% | 0.2857 | 0.2630 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 54 | 35,2% | 1,1% | 0.0556 | 0.0328 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 40 | 35,0% | 0,9% | 0.0500 | 0.0273 |
| RR2_volz_high50 | 1 | vol_z high50@-0.285748 | 57 | 31,6% | -2,5% | -0.0526 | -0.0754 |

