# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_volz_high50 | 1 | vol_z high50@-0.259552 | 50 | 50,0% | 6,3% | 0.0000 | 0.1264 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 42 | 47,6% | 3,9% | -0.0476 | 0.0788 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350671 AND slope_strength low80@0.269444 | 35 | 45,7% | 2,0% | -0.0857 | 0.0407 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.23211 | 48 | 43,8% | 0,1% | -0.1250 | 0.0014 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.21189 | 47 | 42,6% | 6,8% | 0.0638 | 0.1710 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 41 | 39,0% | 3,3% | -0.0244 | 0.0828 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.280266 | 50 | 38,0% | 2,3% | -0.0500 | 0.0571 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_rsi_high60__AND__pullback_atr_low50 | 2 | rsi high60@59.8003 AND pullback_from_new_high_atr low50@1.05693 | 38 | 36,8% | 4,7% | 0.1053 | 0.1423 |
| RR2_rsi_high60 | 1 | rsi high60@59.8003 | 55 | 36,4% | 4,3% | 0.0909 | 0.1279 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350671 | 39 | 35,9% | 3,8% | 0.0769 | 0.1140 |
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05693 | 48 | 35,4% | 3,3% | 0.0625 | 0.0995 |
| RR2_volz_high50 | 1 | vol_z high50@-0.29227 | 49 | 34,7% | 2,6% | 0.0408 | 0.0779 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 55 | 32,7% | 0,6% | -0.0182 | 0.0189 |

