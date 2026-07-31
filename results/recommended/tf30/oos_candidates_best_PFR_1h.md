# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0371925 AND body_pct high60@0.178797 | 44 | 52,3% | 6,3% | 0.0455 | 0.1257 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 63 | 50,8% | 4,8% | 0.0159 | 0.0962 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 82 | 50,0% | 4,0% | 0.0000 | 0.0803 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.27659 | 56 | 50,0% | 4,0% | 0.0000 | 0.0803 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31524 | 66 | 47,0% | 1,0% | -0.0606 | 0.0197 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.274613 | 53 | 47,2% | 8,5% | 0.1792 | 0.2133 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.27288 | 65 | 44,6% | 6,0% | 0.1154 | 0.1495 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 60 | 43,3% | 4,7% | 0.0833 | 0.1174 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.290896 | 82 | 41,5% | 2,8% | 0.0366 | 0.0707 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0456 | 77 | 42,9% | 10,0% | 0.2857 | 0.3013 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 71 | 42,3% | 9,4% | 0.2676 | 0.2832 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 58 | 37,9% | 5,1% | 0.1379 | 0.1536 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 81 | 35,8% | 3,0% | 0.0741 | 0.0897 |
| RR2_volz_high50 | 1 | vol_z high50@-0.302112 | 79 | 32,9% | 0,1% | -0.0127 | 0.0030 |

