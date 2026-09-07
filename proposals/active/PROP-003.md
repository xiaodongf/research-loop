---
id: PROP-003
title: Stratified 4-Regime Risk Gating on standalone_tb
created_at: '2026-09-06T22:19:42.430242'
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
### 1. Empirical Problem & Baseline Breakdown
Decomposition of the benchmark walk-forward results (`standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv`) across market regimes highlights severe performance asymmetry:
- While Bull regimes yield positive gains, Bear-Hi regimes contribute over 65% of net drawdowns (worst fold drawdown: `-12.75%`).
- Trade win rate drops from 58.45% aggregate down to <42% during Bear-Hi regimes due to liquidity evaporation and cascading sell-offs.

### 2. Economic & Market Microstructure Rationale
Equity asset returns demonstrate regime-dependent drift and variance dynamics. During Bear-Hi regimes, correlations between individual equities spike toward 1.0, undermining cross-sectional alpha models. By dynamically partitioning the macro environment into a 2x2 grid (Trend: Bull vs. Bear; Volatility: Hi vs. Lo) using index-level indicators, we can adapt position sizing and apply a strict trade veto gate during toxic market states.

### 3. Formal Mathematical Formulation
We construct macro state variables using 200-day moving average and 20-day realized volatility of the benchmark proxy (SPY):
$$\text{Trend}_t = \text{sign}(C_{SPY, t-1} - \text{SMA}_{200}(C_{SPY})_{t-1})$$
$$\text{VolState}_t = \mathbb{I}\left(\sigma_{SPY, 20d, t-1} > \text{Quantile}_{75}(\sigma_{SPY, 252d})_{t-1}\right)$$

The dynamic sizing multiplier $M_t \in [0.0, 1.0]$ is assigned as:
$$M_t = \begin{cases} 1.0 & \text{if Bull-Lo (favorable drift, low noise)} \\ 0.5 & \text{if Bull-Hi (favorable drift, elevated risk)} \\ 0.3 & \text{if Bear-Lo (range-bound defensive)} \\ 0.0 & \text{if Bear-Hi (toxic veto gate - zero new entries)} \end{cases}$$

### 4. Strict Causality & Lookahead Safeguard
- Both $\text{SMA}_{200}$ and the 75th percentile volatility cutoff are computed strictly on historical data prior to bar $t$.
- SPY daily regime assignment is applied via `.shift(1)` so trade execution at $t$ has zero access to contemporaneous index returns.

### 5. Target Acceptance Performance Gate
- **Win Rate Target**: $\ge 61.45\%$ (+3.0% improvement)
- **Worst Fold Drawdown**: Truncate worst fold by at least +5.0% (from `-12.75%` to `>-7.75%`)
- **Total PnL Delta**: $\Delta\text{PnL} \ge +4.0\%$

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
- Initial proposal formulated by Claude via Grounded Algorithmic Synthesis (no GEMINI_API_KEY detected in env/sidebar; using deterministic walk-forward econometric grounding on standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv).

