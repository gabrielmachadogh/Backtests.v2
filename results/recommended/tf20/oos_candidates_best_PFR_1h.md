# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.238376 | 55 | 47,3% | 3,2% | -0.0545 | 0.0637 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.24209 | 46 | 45,7% | 1,6% | -0.0870 | 0.0313 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 40 | 45,0% | 0,9% | -0.1000 | 0.0183 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372324 AND slope_strength low80@0.270667 | 36 | 44,4% | 0,4% | -0.1111 | 0.0072 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.22051 | 45 | 46,7% | 8,5% | 0.1667 | 0.2116 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.271206 | 35 | 42,9% | 4,7% | 0.0714 | 0.1164 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 38 | 39,5% | 1,3% | -0.0132 | 0.0318 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.259552 | 56 | 39,3% | 1,1% | -0.0179 | 0.0271 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@60.0219 AND pullback_from_new_high_atr low50@1.04786 | 38 | 50,0% | 17,4% | 0.5000 | 0.5233 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.04786 | 49 | 42,9% | 10,3% | 0.2857 | 0.3090 |
| RR2_rsi_high60 | 1 | rsi high60@60.0219 | 54 | 42,6% | 10,0% | 0.2778 | 0.3010 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 37 | 35,1% | 2,6% | 0.0541 | 0.0773 |
| RR2_volz_high50 | 1 | vol_z high50@-0.286218 | 55 | 32,7% | 0,2% | -0.0182 | 0.0051 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 56 | 32,1% | -0,4% | -0.0357 | -0.0125 |

