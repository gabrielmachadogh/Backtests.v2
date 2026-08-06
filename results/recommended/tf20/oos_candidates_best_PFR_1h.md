# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.239186 | 54 | 48,1% | 3,6% | -0.0370 | 0.0717 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 40 | 47,5% | 2,9% | -0.0500 | 0.0587 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.271206 | 37 | 45,9% | 1,4% | -0.0811 | 0.0276 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24094 | 46 | 45,7% | 1,1% | -0.0870 | 0.0217 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22497 | 44 | 45,5% | 8,0% | 0.1364 | 0.1989 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.271171 | 35 | 42,9% | 5,4% | 0.0714 | 0.1339 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 38 | 39,5% | 2,0% | -0.0132 | 0.0493 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.275492 | 56 | 37,5% | 0,0% | -0.0625 | 0.0000 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8816 AND pullback_from_new_high_atr low50@1.05303 | 39 | 48,7% | 16,2% | 0.4615 | 0.4848 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 49 | 42,9% | 10,3% | 0.2857 | 0.3090 |
| RR2_rsi_high60 | 1 | rsi high60@59.8816 | 57 | 40,4% | 7,8% | 0.2105 | 0.2338 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 37 | 35,1% | 2,6% | 0.0541 | 0.0773 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 33,9% | 1,4% | 0.0179 | 0.0411 |
| RR2_volz_high50 | 1 | vol_z high50@-0.288793 | 55 | 32,7% | 0,2% | -0.0182 | 0.0051 |

