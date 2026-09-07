"""Stage 2: Cross-Model Peer Review Routing Tests (test_cross_review.py).

Verifies:
1. Routing proposal from Author Model (e.g. Gemini) to Peer Reviewer (e.g. Claude) -> PEER_REVIEW.
2. Adversarial Red-Team prompt generation for peer auditor.
3. Appending structured peer critique log with verdict without corrupting markdown format.
4. Automatic return to HUMAN_REVIEW with debate thread intact.
5. Multi-round multi-model revision cycle (Gemini -> Claude critique -> GPT revision -> Human approve).
"""
import pytest
from pathlib import Path

from src.core.schema import ProposalStatus, ModelName, StrategyTarget, ReviewVerdict
from src.core.state_machine import ProposalStateMachine, InvalidStateTransitionError
from src.core.proposal_manager import ProposalManager
from src.core.ideation import PromptBuilder


def test_human_route_to_peer(tmp_path):
    pm = ProposalManager(base_dir=tmp_path / "proposals")
    proposal = pm.create_proposal(
        title="Dynamic Volatility Factor",
        author_model=ModelName.GEMINI,
        target_strategy=StrategyTarget.STANDALONE_TB,
        assigned_branch="dev",
        hypothesis="Normalizing features by 20-day Parkinson volatility improves robustness.",
        proposed_files=["data_access/features.py"],
    )
    # Move to HUMAN_REVIEW
    ProposalStateMachine.submit_to_human(proposal)
    assert proposal.metadata.status == ProposalStatus.HUMAN_REVIEW

    # Human routes to Claude for adversarial audit
    ProposalStateMachine.route_to_peer(proposal, reviewer=ModelName.CLAUDE)
    assert proposal.metadata.status == ProposalStatus.PEER_REVIEW
    assert "Assigned for peer review to claude" in proposal.discussion_log


def test_peer_review_prompt_generation(tmp_path):
    pm = ProposalManager(base_dir=tmp_path / "proposals")
    proposal = pm.create_proposal(
        title="Dynamic Volatility Factor",
        author_model=ModelName.GEMINI,
        target_strategy=StrategyTarget.STANDALONE_TB,
        assigned_branch="dev",
        hypothesis="Calculate rolling vol using window=20 on daily high and low.",
        proposed_files=["data_access/features.py"],
        implementation_spec="df['parkinson_vol'] = parkinson(df['high'], df['low'], window=20)",
    )
    prompt = PromptBuilder.create_peer_review_prompt(
        reviewer_model=ModelName.CLAUDE,
        proposal=proposal,
        critique_focus="lookahead bias and causality",
    )

    assert "adversarial Quantitative Peer Reviewer (claude)" in prompt
    assert "lookahead bias and causality" in prompt
    assert "Dynamic Volatility Factor" in prompt
    assert "PROP-001" in prompt
    assert "shift(1)" in prompt


def test_append_peer_critique_log_and_return_to_human(tmp_path):
    pm = ProposalManager(base_dir=tmp_path / "proposals")
    proposal = pm.create_proposal(
        title="Dynamic Volatility Factor",
        author_model=ModelName.GEMINI,
        target_strategy=StrategyTarget.STANDALONE_TB,
        assigned_branch="dev",
        hypothesis="Features normalized by volatility.",
    )
    ProposalStateMachine.submit_to_human(proposal)
    ProposalStateMachine.route_to_peer(proposal, reviewer=ModelName.CLAUDE)

    # Claude records critique
    critique_comment = "Ensure rolling window includes shift(1) to avoid leaking today's close into today's feature vector."
    ProposalStateMachine.add_peer_critique(
        proposal=proposal,
        reviewer=ModelName.CLAUDE,
        verdict=ReviewVerdict.APPROVED_WITH_CAUTION,
        comment=critique_comment,
    )

    # Asserts state machine returned card to HUMAN_REVIEW
    assert proposal.metadata.status == ProposalStatus.HUMAN_REVIEW
    assert len(proposal.metadata.reviewers) == 1
    assert proposal.metadata.reviewers[0].model == ModelName.CLAUDE
    assert proposal.metadata.reviewers[0].verdict == ReviewVerdict.APPROVED_WITH_CAUTION
    assert critique_comment in proposal.discussion_log

    # Persist and verify round-trip serialization preserves critique
    pm.save_proposal(proposal)
    loaded = pm.load_proposal(proposal.id)
    assert loaded.metadata.status == ProposalStatus.HUMAN_REVIEW
    assert len(loaded.metadata.reviewers) == 1
    assert loaded.metadata.reviewers[0].verdict == ReviewVerdict.APPROVED_WITH_CAUTION
    assert critique_comment in loaded.discussion_log


def test_multi_round_revision(tmp_path):
    """Verifies multi-turn debate: Gemini proposes -> Claude critiques -> GPT revises -> Human approves."""
    pm = ProposalManager(base_dir=tmp_path / "proposals")

    # Turn 1: Gemini proposes
    proposal = pm.create_proposal(
        title="Multi-Model Volatility Filter",
        author_model=ModelName.GEMINI,
        target_strategy=StrategyTarget.STANDALONE_TB,
        assigned_branch="dev",
        hypothesis="High volatility regime filter improves precision.",
    )
    ProposalStateMachine.submit_to_human(proposal)

    # Turn 2: Routed to Claude for red-teaming
    ProposalStateMachine.route_to_peer(proposal, reviewer=ModelName.CLAUDE)
    ProposalStateMachine.add_peer_critique(
        proposal=proposal,
        reviewer=ModelName.CLAUDE,
        verdict=ReviewVerdict.NEEDS_REVISION,
        comment="Threshold is too aggressive, reduces trade count by 80%. Consider ATR quantile instead.",
    )
    assert proposal.metadata.status == ProposalStatus.HUMAN_REVIEW

    # Turn 3: Human routes to ChatGPT for revision & synthesis
    ProposalStateMachine.route_to_peer(proposal, reviewer=ModelName.CHATGPT)
    ProposalStateMachine.add_peer_critique(
        proposal=proposal,
        reviewer=ModelName.CHATGPT,
        verdict=ReviewVerdict.APPROVED,
        comment="Revised implementation: uses 70th percentile rolling ATR threshold with shift(1). Trade count preserved.",
    )
    proposal.implementation_spec = "ATR percentile ranking with shift(1)"
    assert proposal.metadata.status == ProposalStatus.HUMAN_REVIEW
    assert len(proposal.metadata.reviewers) == 2

    # Turn 4: Human accepts revised consensus
    ProposalStateMachine.approve(proposal, assigned_branch="dev")
    assert proposal.metadata.status == ProposalStatus.APPROVED
