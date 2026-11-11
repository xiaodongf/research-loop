"""Typed schemas and contracts for the 5-agent Control & Knowledge Bus."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentId(str, Enum):
    MAIN = "main"
    STAGING = "staging"
    DEV = "dev"
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    HUMAN = "human"

    @classmethod
    def normalize(cls, value: str | "AgentId") -> "AgentId":
        if isinstance(value, cls):
            return value
        raw = str(value).strip().lower()
        if raw == "gemini":
            return cls.DEV
        return cls(raw)


WORKSPACE_AGENTS: tuple[AgentId, ...] = (
    AgentId.MAIN,
    AgentId.STAGING,
    AgentId.DEV,
    AgentId.CLAUDE,
    AgentId.CHATGPT,
)


class Intent(str, Enum):
    PROPOSAL_RFC = "PROPOSAL_RFC"
    PEER_CRITIQUE = "PEER_CRITIQUE"
    AUDIT_REQUEST = "AUDIT_REQUEST"
    AUDIT_VERDICT = "AUDIT_VERDICT"
    REMEDIATION_READY = "REMEDIATION_READY"
    INTEGRATION_REQUEST = "INTEGRATION_REQUEST"
    STAGING_RESULT = "STAGING_RESULT"
    BASELINE_UPDATED = "BASELINE_UPDATED"
    KNOWLEDGE_PUBLISHED = "KNOWLEDGE_PUBLISHED"
    ADVISORY_NOTE = "ADVISORY_NOTE"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"


# Decision #1: Only read-only review/audit intents may auto-execute headlessly.
READ_ONLY_AUTO_INTENTS: frozenset[Intent] = frozenset({
    Intent.AUDIT_REQUEST,
    Intent.PROPOSAL_RFC,
    Intent.REMEDIATION_READY,
})

# Decision #5: Reviewer/target addressing is always explicit; no hidden default routing.
EXPLICIT_TARGET_INTENTS: frozenset[Intent] = frozenset({
    Intent.AUDIT_REQUEST,
    Intent.PROPOSAL_RFC,
    Intent.REMEDIATION_READY,
    Intent.INTEGRATION_REQUEST,
})


class ThreadStatus(str, Enum):
    OPEN = "OPEN"
    AWAITING_AUDIT = "AWAITING_AUDIT"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    APPROVED = "APPROVED"
    IN_STAGING = "IN_STAGING"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


class FindingSeverity(str, Enum):
    P0 = "P0"  # Blocking bug, lookahead leakage, or prod/backtest SSOT parity violation
    P1 = "P1"  # Material statistical, edge-case, or operational risk
    P2 = "P2"  # Minor clarity, hygiene, or non-blocking suggestion


class FindingStatus(str, Enum):
    OPEN = "OPEN"
    FIXED_PENDING_VERIFY = "FIXED_PENDING_VERIFY"
    VERIFIED = "VERIFIED"
    WONTFIX = "WONTFIX"


class AuditVerdictType(str, Enum):
    APPROVED = "APPROVED"
    APPROVED_WITH_CAUTION = "APPROVED_WITH_CAUTION"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    NEEDS_REVISION = "NEEDS_REVISION"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"


AGENT_REVIEW_CONTRACTS: Dict[AgentId, Dict[str, str]] = {
    AgentId.MAIN: {
        "title": "Production & Parity Gatekeeper",
        "focus": (
            "1. Production pipeline compatibility (daily_premarket, weekly_validation, live returns).\n"
            "2. Single Source of Truth (StrategyConfig -> RunConfig -> CLI parity; zero hardcoded thresholds).\n"
            "3. Model SHA256 sync integrity and operational safety."
        ),
    },
    AgentId.STAGING: {
        "title": "Pre-Main Full A/A & Parity Integration Gate",
        "focus": (
            "1. Execute sync_from_prod.py --yes, merge origin/main, and run_branch_merge_and_aa_eval.sh --mode full --force-retrain.\n"
            "2. Verify tests/test_config_parity.py passes 100%.\n"
            "3. Emit complete 10-metric pre/post comparative scorecards across DEV, RECENT, and TOTAL."
        ),
    },
    AgentId.DEV: {
        "title": "Implementation & Feature-Set Parity Reviewer",
        "focus": (
            "1. Feature list & column ordering parity (Quantile 27, TB 61, Stacked TB Stage-1 36).\n"
            "2. Training cache fingerprint / invalidation safety.\n"
            "3. Clean minimal diff without unintended side effects."
        ),
    },
    AgentId.CLAUDE: {
        "title": "Adversarial Causality, Parity & Defect Auditor",
        "focus": (
            "1. Strict point-in-time causality (event time vs availability time vs decision time; shift(1) & fold boundary leakage).\n"
            "2. Backtest <-> Production parity (StrategyConfig SSOT, feature transforms, NaN/fallback traps).\n"
            "3. Edge-case regressions, off-by-one indexing, and test coverage."
        ),
    },
    AgentId.CHATGPT: {
        "title": "Econometric, Regime & Cost Challenger",
        "focus": (
            "1. Economic mechanism vs simple exposure reduction (check participation, trade count, and average exposure).\n"
            "2. Regime concentration across Bull/Bear x Hi/Lo vol and DEV vs RECENT consistency.\n"
            "3. Turnover/execution-cost sensitivity and comparison against prior negative experiments."
        ),
    },
}


class CodeContext(BaseModel):
    source_workspace: Optional[str] = None
    branch: str
    base_sha: Optional[str] = None
    head_sha: str
    changed_files: List[str] = Field(default_factory=list)

    @property
    def reviewed_sha(self) -> str:
        return self.head_sha


class FindingInput(BaseModel):
    severity: FindingSeverity = FindingSeverity.P1
    category: str = "general"
    file: str = ""
    lines: str = ""
    problem: str
    suggested_fix: str = ""


class Finding(BaseModel):
    id: str
    thread_id: str
    round_num: int = 1
    author_agent: AgentId
    severity: FindingSeverity
    category: str = "general"
    file: str = ""
    lines: str = ""
    problem: str
    suggested_fix: str = ""
    status: FindingStatus = FindingStatus.OPEN
    resolved_in_sha: Optional[str] = None
    verified_by: Optional[AgentId] = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class MessageEnvelope(BaseModel):
    msg_id: str
    thread_id: str
    round_num: int = 1
    in_reply_to: Optional[str] = None
    from_agent: AgentId
    to_agents: List[AgentId] = Field(default_factory=list)
    cc_agents: List[AgentId] = Field(default_factory=list)
    intent: Intent
    subject: str
    body: str = ""
    code_context: Optional[CodeContext] = None
    verdict: Optional[AuditVerdictType] = None
    findings: List[Finding] = Field(default_factory=list)
    resolved_findings: Dict[str, str] = Field(default_factory=dict)
    verified_findings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    advisories: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("from_agent", mode="before")
    @classmethod
    def _norm_from(cls, v: Any) -> AgentId:
        return AgentId.normalize(v)

    @field_validator("to_agents", "cc_agents", mode="before")
    @classmethod
    def _norm_agent_list(cls, v: Any) -> List[AgentId]:
        if v is None:
            return []
        if isinstance(v, (str, AgentId)):
            raw_items = [v]
        else:
            raw_items = list(v)
        seen: List[AgentId] = []
        for item in raw_items:
            if item is None or str(item).strip() == "":
                continue
            aid = AgentId.normalize(item)
            if aid not in seen:
                seen.append(aid)
        return seen

    @model_validator(mode="after")
    def _validate_explicit_targets(self) -> "MessageEnvelope":
        if self.intent in EXPLICIT_TARGET_INTENTS and not self.to_agents:
            raise ValueError(
                f"Intent '{self.intent.value}' requires explicit non-empty 'to_agents' "
                f"(no implicit/default reviewer routing is permitted)."
            )
        return self


class ThreadRecord(BaseModel):
    thread_id: str
    title: str
    status: ThreadStatus = ThreadStatus.OPEN
    created_by: AgentId
    proposal_id: Optional[str] = None
    latest_round: int = 1
    max_auto_rounds: int = 3
    auto_audit: bool = True
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class WorkClaim(BaseModel):
    claim_id: str
    agent: AgentId
    thread_id: Optional[str] = None
    files: List[str] = Field(default_factory=list)
    summary: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    active: bool = True


class EvalLease(BaseModel):
    lease_id: str
    agent: AgentId
    pid: int
    workspace_path: str = ""
    strategy: str = "standalone_tb"
    mode: str = "quick"
    workers: float = 4.0
    threads_per_worker: float = 6.0
    mem_per_worker_gib: float = 8.0
    thread_id: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    acquired_at: str = Field(default_factory=utc_now_iso)
    released_at: Optional[str] = None
    status: str = "ACTIVE"


class LeaseAcquisitionResult(BaseModel):
    acquired: bool
    lease: Optional[EvalLease] = None
    active_holders: List[EvalLease] = Field(default_factory=list)
    waitlist_position: Optional[int] = None
    wait_id: Optional[str] = None
    capacity_snapshot: Dict[str, Any] = Field(default_factory=dict)
    advisories: List[str] = Field(default_factory=list)
    reason: str = ""


class KnowledgeEntry(BaseModel):
    entry_id: str
    title: str
    category: str  # research_note | experiment_report | ab_positive | ab_negative | ab_neutral
    source_agent: AgentId
    source_branch: str = ""
    commit_sha: Optional[str] = None
    strategy: Optional[str] = None
    outcome: Optional[str] = None
    summary: str
    body_excerpt: str = ""
    tags: List[str] = Field(default_factory=list)
    artifact_path: str = ""
    content_sha256: str
    created_at: str = Field(default_factory=utc_now_iso)


class FleetCheckoutStatus(BaseModel):
    agent: AgentId
    name: str
    workspace_path: str
    exists: bool
    branch: str = "N/A"
    expected_branch: str = ""
    head_sha: str = ""
    head_subject: str = ""
    head_timestamp: str = ""
    is_clean: bool = True
    dirty_count: int = 0
    behind_main: int = 0
    ahead_main: int = 0
    unread_count: int = 0
    active_claims: List[WorkClaim] = Field(default_factory=list)
    advisories: List[str] = Field(default_factory=list)
