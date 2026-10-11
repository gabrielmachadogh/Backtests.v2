# OOS CANDIDATES BEST (PFR 1h)

Regras FIXAS (pré-definidas). Thresholds calculados no TREINO.

## RR 1.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1_ret3_high50__AND__clv_high60 | 2 | ret_3_pct high50@0.0350409 AND clv high60@0.827747 | 45 | 53,3% | 5,8% | 0.0667 | 0.1163 |
| RR1_ret3_high50__AND__bodypct_high60 | 2 | ret_3_pct high50@0.0350409 AND body_pct high60@0.178377 | 47 | 53,2% | 5,7% | 0.0638 | 0.1135 |
| RR1_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 67 | 50,7% | 3,2% | 0.0149 | 0.0646 |
| RR1_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0350409 AND slope_strength low80@0.27717 | 58 | 50,0% | 2,5% | 0.0000 | 0.0496 |
| RR1_volz_high50 | 1 | vol_z high50@-0.277879 | 88 | 48,9% | 1,3% | -0.0227 | 0.0269 |
| RR1_magap_high60 | 1 | ma_gap_pct high60@1.31434 | 66 | 48,5% | 1,0% | -0.0303 | 0.0193 |

## RR 1.5

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR1p5_magap_high60 | 1 | ma_gap_pct high60@1.24667 | 67 | 47,8% | 7,3% | 0.1940 | 0.1830 |
| RR1p5_ret3_high50__AND__slope_low80 | 2 | ret_3_pct high50@0.0361298 AND slope_strength low80@0.275429 | 55 | 45,5% | 5,0% | 0.1364 | 0.1253 |
| RR1p5_ret3_high50 | 1 | ret_3_pct high50@0.0361298 | 64 | 42,2% | 1,7% | 0.0547 | 0.0437 |
| RR1p5_volz_high50 | 1 | vol_z high50@-0.291583 | 88 | 40,9% | 0,5% | 0.0227 | 0.0117 |

## RR 2.0

| name | filters | rule | trades_test | wr_test | Δwr_test | evR_test | ΔevR_test |
|---|---:|---|---:|---:|---:|---:|---:|
| RR2_pullback_atr_low50 | 1 | pullback_from_new_high_atr low50@1.05743 | 76 | 44,7% | 9,1% | 0.3421 | 0.2739 |
| RR2_rsi_high60 | 1 | rsi high60@60.014 | 83 | 44,6% | 9,0% | 0.3373 | 0.2692 |
| RR2_after_new_high_recent_flag_high50 | 1 | after_new_high_recent_flag high50@1 | 82 | 40,2% | 4,6% | 0.2073 | 0.1391 |
| RR2_ret3_high50 | 1 | ret_3_pct high50@0.0350409 | 63 | 38,1% | 2,5% | 0.1429 | 0.0747 |
| RR2_volz_high50 | 1 | vol_z high50@-0.302112 | 84 | 34,5% | -1,1% | 0.0357 | -0.0325 |

