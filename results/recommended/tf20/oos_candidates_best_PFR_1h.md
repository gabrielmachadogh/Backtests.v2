# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23863 | 47 | 46,8% | 2,7% | -0.0638 | 0.0544 |
| RR1_volz_high50 | 1 | vol_z high50@-0.237565 | 54 | 46,3% | 2,2% | -0.0741 | 0.0442 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.271242 | 37 | 45,9% | 1,9% | -0.0811 | 0.0372 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 40 | 45,0% | 0,9% | -0.1000 | 0.0183 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22497 | 44 | 45,5% | 7,3% | 0.1364 | 0.1813 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.271171 | 36 | 44,4% | 6,2% | 0.1111 | 0.1561 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 39 | 41,0% | 2,8% | 0.0256 | 0.0706 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.267522 | 56 | 39,3% | 1,1% | -0.0179 | 0.0271 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0219 AND pullback_from_new_high_atr low50@1.04786 | 38 | 50,0% | 16,7% | 0.5000 | 0.5000 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04786 | 50 | 44,0% | 10,7% | 0.3200 | 0.3200 |
| RR2_rsi_high60 | 1 | rsi high60@60.0219 | 54 | 42,6% | 9,3% | 0.2778 | 0.2778 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 38 | 36,8% | 3,5% | 0.1053 | 0.1053 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 57 | 33,3% | 0,0% | -0.0000 | 0.0000 |
| RR2_volz_high50 | 1 | vol_z high50@-0.286218 | 55 | 32,7% | -0,6% | -0.0182 | -0.0182 |

