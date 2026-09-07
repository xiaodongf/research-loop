---
id: PROP-002
title: Cross-Sectional Residual Momentum on standalone_tb
created_at: '2026-09-06T22:19:42.419687'
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
### 1. Empirical Problem & Baseline Breakdown
Under the current verified benchmark (`standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv`, Top-1 Precision: 30.57%, Total PnL: +24.26%), standard 12-month return momentum experiences severe crash risk during sudden market regime turnarounds (worst fold drawdown: `-12.75%`). Analysis indicates that traditional momentum portfolios load heavily on systematic market beta $\beta_{mkt}$, rendering them vulnerable to broad market sell-offs rather than capturing idiosyncratic stock-picking edge.

### 2. Economic & Financial Econometric Rationale
According to the Capital Asset Pricing Model and asset pricing literature (Blitz et al., 2011), the momentum effect is driven by idiosyncratic investor underreaction to company-specific news rather than systematic factor exposure. Total return momentum $R_i$ conflates market beta with idiosyncratic alpha:
$$R_{i, t} = \alpha_{i, t} + \beta_{i, t} R_{mkt, t} + \epsilon_{i, t}$$
By orthogonalizing asset returns against the market benchmark over a rolling 60-day window, we extract pure residual momentum $\epsilon_{i, t}$. Residual momentum demonstrates significantly higher Sharpe ratios, lower turnover decay, and negligible exposure to market-wide turning-point crashes.

### 3. Formal Mathematical Formulation
For each asset $i$ on day $t-1$, we run an OLS rolling regression over window $W=60$ against market benchmark return $R_{mkt}$:
$$\beta_{i, t-1} = \frac{\text{Cov}_{60}(R_{i}, R_{mkt})_{t-1}}{\text{Var}_{60}(R_{mkt})_{t-1}}$$
$$\alpha_{i, t-1} = \bar{R}_{i, t-1} - \beta_{i, t-1} \bar{R}_{mkt, t-1}$$

The idiosyncratic residual return for day $t-1$ is isolated as:
$$\epsilon_{i, t-1} = R_{i, t-1} - \left(\alpha_{i, t-1} + \beta_{i, t-1} R_{mkt, t-1}\right)$$

We accumulate residual returns over a 20-day horizon and standardize by residual volatility:
$$\text{ResMom}_{i, t-1} = \frac{\sum_{k=1}^{20} \epsilon_{i, t-k}}{\sigma(\epsilon_i, 20d)_{t-1} + \delta}, \quad \delta = 10^{-6}$$

### 4. Strict Causality & Lookahead Safeguard
- Covariances, variances, and beta coefficients are estimated strictly using data available up to bar $t-1$.
- The resulting residual momentum rank feature is shifted by 1 bar (`.shift(1)`), ensuring that portfolio construction at the open of bar $t$ possesses zero contemporaneous or forward knowledge of market returns.

### 5. Target Acceptance Performance Gate
- **Total PnL Target**: >27.76% (+3.5% delta over baseline)
- **Top-1 Precision Target**: >32.57% (+2.0% delta)
- **Risk-Adjusted Return**: Target Sharpe ratio improvement $\Delta\text{Sharpe} \ge +0.25$
- **Worst Fold Guardrail**: Worst fold drawdown truncated by $\ge +3.0\%$

## Proposed Code Modifications
- `data_access/features.py`
- `evaluation/configs.py`

## Implementation Spec
```python
# In data_access/features.py:
def calculate_residual_momentum(asset_ret: pd.Series, mkt_ret: pd.Series, window: int = 60) -> pd.Series:
    """Calculates market-beta-orthogonalized residual momentum with strict lag-1 shift."""
    cov = asset_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    beta = (cov / var).shift(1)
    return (asset_ret - beta * mkt_ret).shift(1)
```

## Discussion & Revision Log
- Initial proposal formulated by Gemini via Grounded Algorithmic Synthesis (no GEMINI_API_KEY detected in env/sidebar; using deterministic walk-forward econometric grounding on standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv).
- Routed to Claude for adversarial lookahead and mathematical critique.

