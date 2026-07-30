# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 53 | 47,2% | 3,2% | -0.0566 | 0.0643 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 39 | 46,2% | 2,2% | -0.0769 | 0.0440 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23863 | 47 | 44,7% | 0,7% | -0.1064 | 0.0145 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0383034 AND slope_strength low80@0.271242 | 36 | 44,4% | 0,5% | -0.1111 | 0.0098 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22051 | 46 | 45,7% | 8,2% | 0.1413 | 0.2038 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0383034 AND slope_strength low80@0.271206 | 35 | 42,9% | 5,4% | 0.0714 | 0.1339 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 38 | 39,5% | 2,0% | -0.0132 | 0.0493 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.277879 | 56 | 37,5% | 0,0% | -0.0625 | 0.0000 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8816 AND pullback_from_new_high_atr low50@1.05303 | 39 | 48,7% | 15,8% | 0.4615 | 0.4733 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 49 | 42,9% | 9,9% | 0.2857 | 0.2975 |
| RR2_rsi_high60 | 1 | rsi high60@59.8816 | 57 | 40,4% | 7,4% | 0.2105 | 0.2223 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 37 | 35,1% | 2,2% | 0.0541 | 0.0658 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 55 | 34,5% | 1,6% | 0.0364 | 0.0481 |
| RR2_volz_high50 | 1 | vol_z high50@-0.288793 | 54 | 33,3% | 0,4% | -0.0000 | 0.0118 |

