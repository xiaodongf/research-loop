"""Data models and schemas for Research Loop RFC proposals, reviews, and evaluation scorecards."""
from __future__ import annotations
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class ProposalStatus(str, Enum):
    DRAFT = "DRAFT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    PEER_REVIEW = "PEER_REVIEW"
    APPROVED = "APPROVED"
    IMPLEMENTING = "IMPLEMENTING"
    AB_TESTING = "AB_TESTING"
    EVALUATED = "EVALUATED"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class ModelName(str, Enum):
    GEMINI = "gemini"
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    HUMAN = "human"


class StrategyTarget(str, Enum):
    STANDALONE_TB = "standalone_tb"
    QUANTILE = "quantile"
    STACKED_TB = "stacked_tb"
    HYBRID = "hybrid"


class ReviewVerdict(str, Enum):
    APPROVED = "APPROVED"
    APPROVED_WITH_CAUTION = "APPROVED_WITH_CAUTION"
    NEEDS_REVISION = "NEEDS_REVISION"
    REJECTED = "REJECTED"


class ReviewEntry(BaseModel):
    model: ModelName
    timestamp: datetime = Field(default_factory=datetime.now)
    verdict: ReviewVerdict
    comment: str


class AcceptanceCriteria(BaseModel):
    eval_mode: str = "quick"  # quick, medium, full
    min_pnl_delta: float = 0.0  # percentage points
    min_win_rate_delta: float = 0.0
    max_drawdown_limit: Optional[float] = None


class ProposalMetadata(BaseModel):
    id: str  # e.g. "PROP-001"
    title: str
    created_at: datetime = Field(default_factory=datetime.now)
    author_model: ModelName
    assigned_branch: str = "claude"  # dev, claude, chatgpt
    status: ProposalStatus = ProposalStatus.HUMAN_REVIEW
    target_strategy: StrategyTarget
    reviewers: List[ReviewEntry] = Field(default_factory=list)
    acceptance_criteria: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    variant_tag: Optional[str] = None
    commit_hash: Optional[str] = None
    rejection_reason: Optional[str] = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not v or not v.startswith("PROP-"):
            raise ValueError(f"Proposal ID must start with 'PROP-', got {v}")
        return v


class Proposal(BaseModel):
    metadata: ProposalMetadata
    hypothesis: str
    proposed_files: List[str] = Field(default_factory=list)
    implementation_spec: str = ""
    discussion_log: str = ""
    scorecard_summary: Optional[Any] = None

    @property
    def evaluation_results(self) -> Optional[Any]:
        return self.scorecard_summary

    @evaluation_results.setter
    def evaluation_results(self, val: Any) -> None:
        self.scorecard_summary = val

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def status(self) -> ProposalStatus:
        return self.metadata.status

    @property
    def author_model(self) -> ModelName:
        return self.metadata.author_model


class FoldMetricRow(BaseModel):
    fold: str
    regime: str
    spy: float
    base_n: int
    base_win: float
    base_total_return: float
    base_max_dd: float
    base_sharpe: float
    base_prec1: float
    var_n: int
    var_win: float
    var_total_return: float
    var_max_dd: float
    var_sharpe: float
    var_prec1: float
    delta_total_return: float
    delta_win: float
    delta_prec1: float


class ScorecardSummary(BaseModel):
    variant: str
    strategy: str
    fold_set: str
    timestamp: str
    git_commit: str
    num_folds: int
    base_pnl_total: float
    var_pnl_total: float
    delta_pnl_total: float
    base_pnl_ex_largest: float
    var_pnl_ex_largest: float
    delta_pnl_ex_largest: float
    base_win_rate: float
    var_win_rate: float
    delta_win_rate: float
    base_prec1_mean: float
    var_prec1_mean: float
    delta_prec1_mean: float
    worst_fold_base: float
    worst_fold_var: float
    fold_wins: int
    regime_deltas: Dict[str, float] = Field(default_factory=dict)
    rows: List[FoldMetricRow] = Field(default_factory=list)

    @property
    def base_top1_precision(self) -> float:
        return self.base_prec1_mean

    @property
    def variant_top1_precision(self) -> float:
        return self.var_prec1_mean

    @property
    def delta_top1_precision(self) -> float:
        return self.delta_prec1_mean

    @property
    def delta_total_pnl(self) -> float:
        return self.delta_pnl_total
