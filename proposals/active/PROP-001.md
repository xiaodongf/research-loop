---
id: PROP-001
title: Multi-Horizon Garman-Klass Volatility Normalization on standalone_tb
created_at: '2026-09-06T22:19:42.397485'
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
Empirical inspection of current verified baseline (`standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv`, Top-1 Precision: 30.57%, Total PnL: +24.26%, Win Rate: 58.45%) demonstrates severe vulnerability to volatility clustering. In particular, the worst-performing walk-forward fold suffered a maximum drawdown of `-12.75%`. Examination of per-trade records reveals that false-positive breakout entries concentrate heavily during sudden intraday volatility spikes, where static thresholds fail to adapt to expanding price ranges.

### 2. Economic & Market Microstructure Rationale
Static return thresholds assume homoskedastic price distributions across time. However, asset returns exhibit pronounced volatility clustering (Mandelbrot, 1963; Engle, 1982). Using close-to-close returns alone ignores critical intra-bar extreme excursions. The Garman-Klass volatility estimator (Garman & Klass, 1980) incorporates high, low, open, and close prices, providing an estimator that is up to 7.4 times more statistically efficient than standard close-to-close variance. By standardizing raw price signals by their local Garman-Klass volatility, we isolate genuine directional momentum from market noise and wild whipsaws.

### 3. Formal Mathematical Formulation
For each asset at bar $t$, we compute the 20-day rolling Garman-Klass variance:
$$\sigma_{GK, t}^2 = 0.5 \left(\ln\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_t}{O_t}\right)^2$$
$$\sigma_{GK, t} = \sqrt{\frac{1}{20}\sum_{k=0}^{19} \sigma_{GK, t-k}^2}$$

The volatility-normalized feature score $F_{norm, t}$ is defined as:
$$F_{norm, t} = \frac{R_{10d, t-1}}{\sigma_{GK, t-1} + \epsilon}, \quad \epsilon = 10^{-6}$$

### 4. Strict Causality & Lookahead Safeguard
- The rolling estimator $\sigma_{GK, t-1}$ strictly consumes price data up to bar $t-1$.
- The resulting feature series is explicitly shifted by 1 bar (`.shift(1)`), ensuring that the entry decision at the open of bar $t$ relies exclusively on information known prior to market open.
- Evaluation fold boundaries in `evaluation/folds.py` remain strictly untouched and immutable.

### 5. Target Acceptance Performance Gate
- **Top-1 Precision Target**: >32.57% (+2.0% delta over current baseline)
- **Total PnL Target**: >29.26% (+5.0% cumulative outperformance)
- **Worst Fold Guardrail**: Drawdown truncated to >-9.75%

## Proposed Code Modifications
- `data_access/features.py`
- `evaluation/configs.py`

## Implementation Spec
```python
# 1. In data_access/features.py:
def calculate_garman_klass_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculates 20-day rolling Garman-Klass volatility with strict lag-1 shift."""
    log_hl = np.log(df['high'] / df['low']) ** 2
    log_co = np.log(df['close'] / df['open']) ** 2
    var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(var.rolling(window=window).mean()).shift(1)

# 2. In evaluation/configs.py:
# Add use_gk_vol_normalization: bool = True to RunConfig
```

## Discussion & Revision Log
- Initial proposal formulated by Gemini via Grounded Algorithmic Synthesis (no GEMINI_API_KEY detected in env/sidebar; using deterministic walk-forward econometric grounding on standalone_tb_ab-ml_dynamic_supervisor_QUICK_20260830_212735_98dbd4a.csv).
- Routed to Claude for adversarial lookahead and mathematical critique.

