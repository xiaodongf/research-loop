---
id: PROP-001
title: Multi-Horizon Garman-Klass Volatility Normalization on standalone_tb
created_at: '2026-09-06T17:21:21.678902'
author_model: gemini
assigned_branch: dev
status: PEER_REVIEW
target_strategy: standalone_tb
reviewers:
- {}
acceptance_criteria:
  eval_mode: quick
  min_pnl_delta: 2.0
  min_win_rate_delta: 0.0
  max_drawdown_limit: null
variant_tag: null
commit_hash: null
rejection_reason: null
---

## Hypothesis
Empirical inspection of current baseline (standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv, Top-1 Precision: 30.57%, Total PnL: +24.26%) reveals significant false-positive breakout drawdowns in high-volatility regimes (worst fold: -12.75%).

We propose normalizing price return features by 20-day Garman-Klass volatility:
$$\sigma_{GK}^2 = 0.5 \left(\ln\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_t}{O_t}\right)^2$$

Safeguard: To strictly preserve causality and eliminate lookahead bias, the rolling estimator is shifted by 1 bar:
$$F_{norm, t} = \frac{R_t}{\sigma_{GK, t-1} + \epsilon}$$

Target Improvement:
- Precision Target: >32.57% (+2.0% delta over baseline)
- Total PnL Target: >29.26% (+5.0% delta over baseline)
- Worst Fold Threshold: >-9.75%

## Proposed Code Modifications
- `data_access/features.py`
- `evaluation/configs.py`

## Implementation Spec
```python
# 1. In data_access/features.py:
def calculate_garman_klass_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    log_hl = np.log(df['high'] / df['low']) ** 2
    log_co = np.log(df['close'] / df['open']) ** 2
    var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(var.rolling(window=window).mean()).shift(1)

# 2. In evaluation/configs.py:
# Add use_gk_vol_normalization: bool = True to RunConfig
```

## Discussion & Revision Log
- Handed off to Claude for mathematical and lookahead audit.

