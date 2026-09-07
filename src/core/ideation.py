"""Proposal Ideation Engine: context assembly and prompt building for grounded algorithmic RFCs."""
from __future__ import annotations
import os, glob
from pathlib import Path
from typing import Optional, List, Dict, Any
from src.core.schema import ModelName, StrategyTarget


class ContextBuilder:
    @staticmethod
    def get_strategy_overview(strategy: StrategyTarget) -> str:
        overviews = {
            StrategyTarget.STANDALONE_TB: (
                "Strategy: Standalone Triple-Barrier (standalone_tb)\n"
                "- Entry logic: Top-K ranking by probability threshold (default top_k=1, threshold=0.73).\n"
                "- Primary metrics: Top-1 Precision@10%, Total PnL, Trade Win Rate, Worst-fold guardrail.\n"
                "- Key files: strategies/core_strategy/, data_access/features.py, evaluation/configs.py"
            ),
            StrategyTarget.QUANTILE: (
                "Strategy: Standalone Quantile Ranker (quantile)\n"
                "- Entry logic: Daily quantile classification & XGBoost ranker.\n"
                "- Primary metrics: Top-1 Precision, PnL, Win Rate.\n"
                "- Key files: ml/models/xgboost_model.py, config.py, train.py"
            ),
            StrategyTarget.STACKED_TB: (
                "Strategy: Stacked Triple-Barrier (stacked_tb)\n"
                "- Entry logic: Multi-model tiered ranking with dynamic volatility and regime filtering.\n"
                "- Key files: evaluation/strategies/standalone_tb.py, evaluation/hazard_supervisor.py"
            ),
            StrategyTarget.HYBRID: (
                "Strategy: Hybrid Model (hybrid)\n"
                "- Entry logic: Combined Chronos-2 time-series and XGBoost tabular signals."
            ),
        }
        return overviews.get(strategy, f"Strategy: {strategy.value}")

    @staticmethod
    def get_recent_baseline_summary(results_dir: Path) -> str:
        if not results_dir.exists():
            return "No prior evaluation CSVs found; golden truth baseline from main branch applies."
        
        csv_files = sorted(results_dir.glob("*.csv"), key=os.path.getmtime, reverse=True)
        if not csv_files:
            return "No prior evaluation CSVs found; golden truth baseline from main branch applies."
        
        # Take the most recent CSV
        latest = csv_files[0]
        return f"Recent Benchmark CSV: {latest.name}"

    @staticmethod
    def get_negative_history(rejected_dir: Path) -> str:
        if not rejected_dir.exists():
            return "No past rejected proposals recorded."
        
        rejected_files = sorted(rejected_dir.glob("PROP-*.md"))
        if not rejected_files:
            return "No past rejected proposals recorded."
        
        summaries = []
        for f in rejected_files[:5]:
            summaries.append(f"- {f.stem}: Consult {f.name} for rejected hypothesis and failure reasons.")
        return "\n".join(summaries)

    @classmethod
    def build(cls, strategy: StrategyTarget, results_dir: Optional[Path] = None, rejected_dir: Optional[Path] = None) -> str:
        res_dir = results_dir or Path("/home/xiaodong/workspace/trading/evaluation/outputs/results")
        rej_dir = rejected_dir or Path("proposals/rejected")

        parts = [
            "### 1. Strategy Context & Target Components",
            cls.get_strategy_overview(strategy),
            "\n### 2. Baseline Performance Context",
            cls.get_recent_baseline_summary(res_dir),
            "\n### 3. Negative Knowledge Base (Do Not Repeat Failed Hypotheses)",
            cls.get_negative_history(rej_dir),
        ]
        return "\n".join(parts)


class PromptBuilder:
    @staticmethod
    def create_ideation_prompt(
        author_model: ModelName,
        strategy: StrategyTarget,
        theme: str,
        context: str,
        next_id: str = "PROP-001"
    ) -> str:
        return f"""You are acting as an expert quantitative ML researcher ({author_model.value}).
Your objective is to propose ONE concrete, mathematically motivated, testable improvement to the trading system.

Research Theme: {theme}
Target Strategy: {strategy.value}

Below is the verified system context:
{context}

STRICT REQUIREMENTS:
1. You must output a structured RFC proposal starting with YAML frontmatter delimited by '---'.
2. The proposal ID MUST be '{next_id}'.
3. Do not modify core evaluation fold logic (evaluation/folds.py is strictly immutable).
4. Propose changes that can be evaluated using: python -m evaluation.cli --strategy {strategy.value} --mode quick --ab <variant_name>

Required Output Template:
---
id: {next_id}
title: "<Concise Descriptive Title>"
author_model: {author_model.value}
assigned_branch: "{'claude' if author_model == ModelName.CLAUDE else ('dev' if author_model == ModelName.GEMINI else 'chatgpt')}"
status: HUMAN_REVIEW
target_strategy: {strategy.value}
variant_tag: "<short_variant_name>"
acceptance_criteria:
  eval_mode: quick
  min_pnl_delta: 2.0
  min_win_rate_delta: 0.0
---

## Hypothesis
<Detailed theoretical and quantitative explanation of why this change improves out-of-sample performance>

## Proposed Code Modifications
- `data_access/features.py`
- `evaluation/configs.py`

## Implementation Spec
<Step-by-step code change specification>

## Discussion & Revision Log
- Initial proposal drafted by {author_model.value}.
"""

    @staticmethod
    def get_next_model_in_rotation(current_model: ModelName) -> ModelName:
        """Round-robin model rotation order: Gemini -> Claude -> ChatGPT -> Gemini."""
        rotation = {
            ModelName.GEMINI: ModelName.CLAUDE,
            ModelName.CLAUDE: ModelName.CHATGPT,
            ModelName.CHATGPT: ModelName.GEMINI,
        }
        return rotation.get(current_model, ModelName.GEMINI)

    @staticmethod
    def get_thematic_assignment(theme: str) -> ModelName:
        """Maps research theme/directive to specialized frontier model."""
        theme_lower = theme.lower()
        if any(k in theme_lower for k in ["lookahead", "leakage", "audit", "math", "causality", "refactor"]):
            return ModelName.CLAUDE
        elif any(k in theme_lower for k in ["feature", "indicator", "signal", "volatility", "alpha"]):
            return ModelName.GEMINI
        else:
            return ModelName.CHATGPT

    @staticmethod
    def create_peer_review_prompt(
        reviewer_model: ModelName,
        proposal: Proposal,
        critique_focus: str = "lookahead bias, causality, and mathematical validity"
    ) -> str:
        """Constructs an adversarial red-team peer review prompt."""
        mod_files = "\n".join(f"- {f}" for f in proposal.proposed_files) if proposal.proposed_files else "None specified."
        return f"""You are acting as an adversarial Quantitative Peer Reviewer ({reviewer_model.value}).
Your task is to RED-TEAM and critically audit the algorithmic proposal below.

Critique Focus: {critique_focus}

Proposal Details:
- ID: {proposal.metadata.id}
- Title: {proposal.metadata.title}
- Author Model: {proposal.metadata.author_model.value}
- Target Strategy: {proposal.metadata.target_strategy.value}
- Proposed Files:
{mod_files}

Hypothesis:
{proposal.hypothesis}

Implementation Spec:
{proposal.implementation_spec}

CRITIQUE GUIDELINES:
1. Lookahead Bias & Causality: Does any calculation use future data or unshifted rolling windows (shift(1) missing)?
2. Overfitting & Snooping: Is this hypothesis robust across both Bull/Bear and Hi/Lo vol regimes?
3. Implementation Feasibility: Can this change be tested directly via python -m evaluation.cli?

Provide your verdict (APPROVED_WITH_CAUTION | NEEDS_REVISION | REJECTED) and concise reasoning.
"""
