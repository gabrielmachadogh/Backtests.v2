# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0372722 AND body_pct high60@0.178278 | 43 | 53,5% | 7,5% | 0.0698 | 0.1501 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 62 | 51,6% | 5,6% | 0.0323 | 0.1126 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.277751 | 55 | 50,9% | 4,9% | 0.0182 | 0.0985 |
| RR1_volz_high50 | 1 | vol_z high50@-0.275492 | 81 | 50,6% | 4,6% | 0.0123 | 0.0926 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31327 | 68 | 45,6% | -0,4% | -0.0882 | -0.0079 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0393346 AND slope_strength low80@0.276009 | 52 | 48,1% | 9,4% | 0.2019 | 0.2360 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24529 | 68 | 44,1% | 5,5% | 0.1029 | 0.1370 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0393346 | 59 | 44,1% | 5,4% | 0.1017 | 0.1358 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.290896 | 81 | 42,0% | 3,3% | 0.0494 | 0.0835 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60 | 1 | rsi high60@60.0535 | 77 | 44,2% | 10,3% | 0.3247 | 0.3089 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 71 | 43,7% | 9,8% | 0.3099 | 0.2941 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0372324 | 58 | 39,7% | 5,8% | 0.1897 | 0.1739 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 78 | 37,2% | 3,3% | 0.1154 | 0.0996 |
| RR2_volz_high50 | 1 | vol_z high50@-0.302112 | 77 | 35,1% | 1,2% | 0.0519 | 0.0362 |

