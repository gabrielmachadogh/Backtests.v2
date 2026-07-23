# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 40 | 47,5% | 3,5% | -0.0500 | 0.0709 |
| RR1_volz_high50 | 1 | vol_z high50@-0.249369 | 53 | 47,2% | 3,2% | -0.0566 | 0.0643 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.271314 | 37 | 45,9% | 2,0% | -0.0811 | 0.0398 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24461 | 47 | 44,7% | 0,7% | -0.1064 | 0.0145 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.23498 | 45 | 44,4% | 6,9% | 0.1111 | 0.1736 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.271278 | 36 | 44,4% | 6,9% | 0.1111 | 0.1736 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 39 | 41,0% | 3,5% | 0.0256 | 0.0881 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.277879 | 55 | 36,4% | -1,1% | -0.0909 | -0.0284 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8003 AND pullback_from_new_high_atr low50@1.05303 | 39 | 48,7% | 15,8% | 0.4615 | 0.4733 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 49 | 42,9% | 9,9% | 0.2857 | 0.2975 |
| RR2_rsi_high60 | 1 | rsi high60@59.8003 | 57 | 40,4% | 7,4% | 0.2105 | 0.2223 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 36 | 36,1% | 3,2% | 0.0833 | 0.0951 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 55 | 34,5% | 1,6% | 0.0364 | 0.0481 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 54 | 33,3% | 0,4% | -0.0000 | 0.0118 |

