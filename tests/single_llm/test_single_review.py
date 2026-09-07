import pytest
from src.core.schema import (
    Proposal, ProposalMetadata, ProposalStatus, ModelName, StrategyTarget
)
from src.core.proposal_manager import ProposalManager
from src.core.state_machine import StateMachine, InvalidStateTransitionError


def create_sample_proposal(status=ProposalStatus.DRAFT, pid="PROP-001"):
    meta = ProposalMetadata(
        id=pid,
        title="Sample Test Proposal",
        author_model=ModelName.CLAUDE,
        assigned_branch="claude",
        status=status,
        target_strategy=StrategyTarget.STANDALONE_TB
    )
    return Proposal(
        metadata=meta,
        hypothesis="Test hypothesis for validation.",
        proposed_files=["data_access/features.py"]
    )


def test_single_draft_to_human_review():
    proposal = create_sample_proposal(status=ProposalStatus.DRAFT)
    assert proposal.status == ProposalStatus.DRAFT

    updated = StateMachine.submit_to_human(proposal)
    assert updated.status == ProposalStatus.HUMAN_REVIEW


def test_single_human_approval():
    proposal = create_sample_proposal(status=ProposalStatus.HUMAN_REVIEW)
    
    approved = StateMachine.approve(proposal, assigned_branch="claude")
    assert approved.status == ProposalStatus.APPROVED
    assert approved.metadata.assigned_branch == "claude"

    # From APPROVED, can start implementation
    implementing = StateMachine.start_implementation(approved)
    assert implementing.status == ProposalStatus.IMPLEMENTING


def test_single_human_rejection_and_file_relocation(tmp_path):
    pm = ProposalManager(root_dir=tmp_path)
    proposal = create_sample_proposal(status=ProposalStatus.HUMAN_REVIEW, pid="PROP-002")
    
    # Save in active dir first
    pm.save_proposal(proposal)
    assert (pm.active_dir / "PROP-002.md").exists()
    assert not (pm.rejected_dir / "PROP-002.md").exists()

    # Reject proposal
    rejected = StateMachine.reject(proposal, reason="Risk of lookahead bias in rolling window.")
    assert rejected.status == ProposalStatus.REJECTED
    assert rejected.metadata.rejection_reason == "Risk of lookahead bias in rolling window."

    # Save should now relocate file to rejected_dir
    saved_path = pm.save_proposal(rejected)
    assert saved_path == pm.rejected_dir / "PROP-002.md"
    assert (pm.rejected_dir / "PROP-002.md").exists()
    assert not (pm.active_dir / "PROP-002.md").exists()


def test_single_state_machine_invariants():
    # Invariant 1: Cannot jump from DRAFT directly to APPROVED
    p_draft = create_sample_proposal(status=ProposalStatus.DRAFT)
    with pytest.raises(InvalidStateTransitionError):
        StateMachine.approve(p_draft)

    # Invariant 2: Cannot transition out of REJECTED
    p_rejected = create_sample_proposal(status=ProposalStatus.REJECTED)
    with pytest.raises(InvalidStateTransitionError):
        StateMachine.approve(p_rejected)
    with pytest.raises(InvalidStateTransitionError):
        StateMachine.submit_to_human(p_rejected)

    # Invariant 3: Cannot transition out of PROMOTED
    p_promoted = create_sample_proposal(status=ProposalStatus.PROMOTED)
    with pytest.raises(InvalidStateTransitionError):
        StateMachine.reject(p_promoted, reason="too late")
