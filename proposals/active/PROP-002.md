---
id: PROP-002
title: Cross-Sectional Residual Momentum on standalone_tb
created_at: '2026-09-06T17:21:21.686613'
author_model: gemini
assigned_branch: dev
status: HUMAN_REVIEW
target_strategy: standalone_tb
reviewers: []
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
Standard momentum ranking exhibits high tail decay during market turning points. Grounded against recent baseline standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv (Top-1 Precision: 30.57%, Total PnL: +24.26%), we propose orthogonalizing 12-month return momentum against 20-day market beta:
$$R_{asset, t} = \alpha_t + \beta_t R_{mkt, t} + \epsilon_t$$

Ranking solely on idiosyncratic residual momentum $\epsilon_t$ isolates true alpha.
Target Metrics: $\Delta\text{PnL} \ge +3.5\%$, $\Delta\text{Sharpe} \ge +0.25$.

## Proposed Code Modifications
- `data_access/features.py`
- `evaluation/configs.py`

## Implementation Spec
```python
# In data_access/features.py:
def calculate_residual_momentum(asset_ret: pd.Series, mkt_ret: pd.Series, window: int = 60) -> pd.Series:
    cov = asset_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    beta = (cov / var).shift(1)
    return (asset_ret - beta * mkt_ret).shift(1)
```

## Discussion & Revision Log
- Initial proposal formulated by Gemini grounded on benchmark standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv.

