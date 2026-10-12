# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0350147 AND clv high60@0.826255 | 46 | 52,2% | 5,4% | 0.0435 | 0.1073 |
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0350147 AND body_pct high60@0.178797 | 48 | 52,1% | 5,3% | 0.0417 | 0.1055 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 68 | 50,0% | 3,2% | 0.0000 | 0.0638 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350147 AND slope_strength low80@0.27659 | 59 | 49,2% | 2,3% | -0.0169 | 0.0469 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31273 | 66 | 48,5% | 1,7% | -0.0303 | 0.0335 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 88 | 47,7% | 0,9% | -0.0455 | 0.0184 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.25839 | 66 | 48,5% | 8,0% | 0.2121 | 0.2011 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.274613 | 56 | 44,6% | 4,2% | 0.1161 | 0.1050 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 65 | 41,5% | 1,1% | 0.0385 | 0.0274 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.290896 | 88 | 40,9% | 0,5% | 0.0227 | 0.0117 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 76 | 43,4% | 8,6% | 0.3026 | 0.2572 |
| RR2_rsi_high60 | 1 | rsi high60@60.0298 | 83 | 43,4% | 8,5% | 0.3012 | 0.2558 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 82 | 39,0% | 4,2% | 0.1707 | 0.1253 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350147 | 64 | 37,5% | 2,7% | 0.1250 | 0.0795 |
| RR2_volz_high50 | 1 | vol_z high50@-0.293428 | 84 | 33,3% | -1,5% | -0.0000 | -0.0455 |

