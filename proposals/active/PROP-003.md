---
id: PROP-003
title: Stratified 4-Regime Risk Gating on standalone_tb
created_at: '2026-09-06T17:21:21.693185'
author_model: claude
assigned_branch: claude
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
Analysis of benchmark baseline (standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv, Win Rate: 58.45%) demonstrates that over 65% of net loss stems from Bear-Hi volatility regimes (worst fold drawdown: -12.75%).

We introduce dynamic regime stratification combining 20-day SPY trend and 20-day realized volatility quantile:
- Bull-Lo: Full position sizing (1.0x)
- Bull-Hi: Half position sizing (0.5x)
- Bear-Lo: Defensively constrained sizing (0.3x)
- Bear-Hi: Trade veto gate (0.0x sizing)

Target Metric Delta: $\Delta\text{Win Rate} \ge +3.0\%$, $\Delta\text{Worst Fold} \ge +5.0\%$.

## Proposed Code Modifications
- `evaluation/strategies/base_strategy.py`
- `evaluation/configs.py`

## Implementation Spec
```python
# In evaluation/strategies/base_strategy.py:
def apply_regime_gating(signal: pd.Series, regime: str) -> pd.Series:
    multipliers = {'Bull-Lo': 1.0, 'Bull-Hi': 0.5, 'Bear-Lo': 0.3, 'Bear-Hi': 0.0}
    return signal * multipliers.get(regime, 1.0)
```

## Discussion & Revision Log
- Initial proposal formulated by Claude grounded on benchmark standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv.

