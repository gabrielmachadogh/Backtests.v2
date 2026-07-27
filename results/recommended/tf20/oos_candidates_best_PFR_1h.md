# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 39 | 48,7% | 3,7% | -0.0256 | 0.0733 |
| RR1_volz_high50 | 1 | vol_z high50@-0.239186 | 52 | 48,1% | 3,0% | -0.0385 | 0.0604 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.271278 | 36 | 47,2% | 2,2% | -0.0556 | 0.0433 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24209 | 47 | 44,7% | -0,4% | -0.1064 | -0.0075 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.271242 | 35 | 45,7% | 7,1% | 0.1429 | 0.1769 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22747 | 45 | 44,4% | 5,8% | 0.1111 | 0.1452 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 38 | 42,1% | 3,5% | 0.0526 | 0.0867 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.275492 | 55 | 38,2% | -0,5% | -0.0455 | -0.0114 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8003 AND pullback_from_new_high_atr low50@1.05303 | 39 | 48,7% | 15,8% | 0.4615 | 0.4733 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 49 | 42,9% | 9,9% | 0.2857 | 0.2975 |
| RR2_rsi_high60 | 1 | rsi high60@59.8003 | 57 | 40,4% | 7,4% | 0.2105 | 0.2223 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 36 | 36,1% | 3,2% | 0.0833 | 0.0951 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 55 | 34,5% | 1,6% | 0.0364 | 0.0481 |
| RR2_volz_high50 | 1 | vol_z high50@-0.291583 | 54 | 33,3% | 0,4% | -0.0000 | 0.0118 |

