"""State machine for proposal lifecycle transitions and governance invariants."""
from __future__ import annotations
from typing import Optional, Dict, Any
from src.core.schema import Proposal, ProposalStatus, ModelName, ReviewVerdict, ReviewEntry


class InvalidStateTransitionError(Exception):
    """Raised when an illegal lifecycle transition is attempted."""
    pass


class StateMachine:
    @staticmethod
    def transition(proposal: Proposal, target_status: ProposalStatus, reason: Optional[str] = None) -> Proposal:
        current = proposal.metadata.status

        if current in (ProposalStatus.PROMOTED, ProposalStatus.REJECTED):
            raise InvalidStateTransitionError(
                f"Cannot transition proposal {proposal.id} out of terminal state {current}."
            )

        # Valid transition rules
        valid_transitions = {
            ProposalStatus.DRAFT: [ProposalStatus.HUMAN_REVIEW],
            ProposalStatus.HUMAN_REVIEW: [ProposalStatus.PEER_REVIEW, ProposalStatus.APPROVED, ProposalStatus.REJECTED],
            ProposalStatus.PEER_REVIEW: [ProposalStatus.HUMAN_REVIEW, ProposalStatus.PEER_REVIEW, ProposalStatus.REJECTED],
            ProposalStatus.APPROVED: [ProposalStatus.IMPLEMENTING, ProposalStatus.REJECTED],
            ProposalStatus.IMPLEMENTING: [ProposalStatus.AB_TESTING, ProposalStatus.HUMAN_REVIEW, ProposalStatus.REJECTED],
            ProposalStatus.AB_TESTING: [ProposalStatus.EVALUATED, ProposalStatus.HUMAN_REVIEW],
            ProposalStatus.EVALUATED: [ProposalStatus.PROMOTED, ProposalStatus.REJECTED, ProposalStatus.HUMAN_REVIEW, ProposalStatus.AB_TESTING],
        }

        allowed = valid_transitions.get(current, [])
        if target_status not in allowed:
            raise InvalidStateTransitionError(
                f"Illegal transition for proposal {proposal.id}: {current} -> {target_status}. Allowed: {allowed}"
            )

        proposal.metadata.status = target_status
        if target_status == ProposalStatus.REJECTED and reason:
            proposal.metadata.rejection_reason = reason

        return proposal

    @classmethod
    def submit_to_human(cls, proposal: Proposal) -> Proposal:
        return cls.transition(proposal, ProposalStatus.HUMAN_REVIEW)

    @classmethod
    def route_to_peer(cls, proposal: Proposal, reviewer: ModelName) -> Proposal:
        cls.transition(proposal, ProposalStatus.PEER_REVIEW)
        # Note reviewer assignment can be logged in discussion
        proposal.discussion_log += f"\n- Assigned for peer review to {reviewer.value}."
        return proposal

    @classmethod
    def add_peer_critique(cls, proposal: Proposal, reviewer: ModelName, verdict: ReviewVerdict, comment: str) -> Proposal:
        if proposal.metadata.status != ProposalStatus.PEER_REVIEW:
            raise InvalidStateTransitionError(
                f"Cannot add peer critique to proposal {proposal.id} in state {proposal.metadata.status}; must be in PEER_REVIEW."
            )
        entry = ReviewEntry(model=reviewer, verdict=verdict, comment=comment)
        proposal.metadata.reviewers.append(entry)
        proposal.discussion_log += f"\n- **{reviewer.value}** ({verdict.value}): {comment}"
        # Returns automatically to HUMAN_REVIEW for human decision
        return cls.transition(proposal, ProposalStatus.HUMAN_REVIEW)

    @classmethod
    def approve(cls, proposal: Proposal, assigned_branch: Optional[str] = None) -> Proposal:
        if assigned_branch:
            proposal.metadata.assigned_branch = assigned_branch
        return cls.transition(proposal, ProposalStatus.APPROVED)

    @classmethod
    def reject(cls, proposal: Proposal, reason: str) -> Proposal:
        return cls.transition(proposal, ProposalStatus.REJECTED, reason=reason)

    @classmethod
    def start_implementation(cls, proposal: Proposal) -> Proposal:
        return cls.transition(proposal, ProposalStatus.IMPLEMENTING)

    @classmethod
    def complete_implementation(cls, proposal: Proposal, commit_hash: str) -> Proposal:
        proposal.metadata.commit_hash = commit_hash
        return cls.transition(proposal, ProposalStatus.AB_TESTING)

    @classmethod
    def complete_evaluation(cls, proposal: Proposal, scorecard: Dict[str, Any]) -> Proposal:
        proposal.scorecard_summary = scorecard
        return cls.transition(proposal, ProposalStatus.EVALUATED)

    @classmethod
    def promote(cls, proposal: Proposal) -> Proposal:
        return cls.transition(proposal, ProposalStatus.PROMOTED)


ProposalStateMachine = StateMachine
