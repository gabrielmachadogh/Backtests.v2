# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0371925 | 62 | 56,5% | 8,7% | 0.1290 | 0.1745 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0371925 AND slope_strength low80@0.28011 | 55 | 56,4% | 8,6% | 0.1273 | 0.1727 |
| RR1_volz_high50 | 1 | vol_z high50@-0.280266 | 77 | 54,5% | 6,8% | 0.0909 | 0.1364 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31327 | 69 | 46,4% | -1,4% | -0.0725 | -0.0270 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0372722 AND slope_strength low80@0.278776 | 52 | 51,9% | 12,6% | 0.2981 | 0.3138 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0372722 | 59 | 47,5% | 8,1% | 0.1864 | 0.2022 |
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24529 | 69 | 44,9% | 5,6% | 0.1232 | 0.1389 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.29227 | 77 | 44,2% | 4,8% | 0.1039 | 0.1196 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0383034 | 56 | 42,9% | 8,7% | 0.2857 | 0.2613 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05303 | 69 | 42,0% | 7,9% | 0.2609 | 0.2365 |
| RR2_rsi_high60 | 1 | rsi high60@60.0961 | 74 | 41,9% | 7,7% | 0.2568 | 0.2324 |
| RR2_volz_high50 | 1 | vol_z high50@-0.319788 | 75 | 37,3% | 3,2% | 0.1200 | 0.0956 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 79 | 35,4% | 1,3% | 0.0633 | 0.0389 |

