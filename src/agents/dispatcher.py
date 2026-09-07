"""Agent Workspace Dispatcher: coordinates in-process execution and automatic evaluation."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Optional, Dict, Any

from src.core.schema import Proposal, ProposalStatus, ScorecardSummary
from src.core.state_machine import ProposalStateMachine
from src.core.proposal_manager import ProposalManager
from src.core.isolation_guard import WorkspaceIsolationGuard
from src.core.git_manager import GitManager
from src.agents.base import AgentRunner, AgentExecutionResult
from src.agents.claude_runner import ClaudeCodeRunner
from src.agents.gemini_runner import GeminiRunner
from src.agents.gpt_runner import GPTRunner
from src.harness.eval_bridge import EvalBridge


class AgentDispatcher:
    """Dispatches approved proposals to the corresponding workspace agent runner."""

    def __init__(
        self,
        config: Dict[str, Any],
        proposal_manager: Optional[ProposalManager] = None,
        eval_bridge: Optional[EvalBridge] = None,
    ):
        self.config = config
        self.proposal_manager = proposal_manager
        self.eval_bridge = eval_bridge

        # Extract workspace paths
        self.workspaces = {}
        ws_conf = config.get("workspaces", {})
        for k, v in ws_conf.items():
            self.workspaces[k] = Path(v.get("path", "")).resolve()

        self.isolation_guard = WorkspaceIsolationGuard(self.workspaces) if self.workspaces else None
        self.git_manager = GitManager(self.workspaces) if self.workspaces else None

        # Runners map
        self.runners: Dict[str, AgentRunner] = {}
        self._init_runners()

    def _init_runners(self):
        ws_conf = self.config.get("workspaces", {})

        if "claude" in ws_conf:
            c_conf = ws_conf["claude"]
            self.runners["claude"] = ClaudeCodeRunner(
                workspace_path=Path(c_conf["path"]),
                binary_path=c_conf.get("binary"),
                isolation_guard=self.isolation_guard,
                git_manager=self.git_manager,
            )

        if "gemini" in ws_conf:
            g_conf = ws_conf["gemini"]
            self.runners["gemini"] = GeminiRunner(
                workspace_path=Path(g_conf["path"]),
                command=g_conf.get("command"),
                isolation_guard=self.isolation_guard,
                git_manager=self.git_manager,
            )

        if "chatgpt" in ws_conf:
            gpt_conf = ws_conf["chatgpt"]
            self.runners["chatgpt"] = GPTRunner(
                workspace_path=Path(gpt_conf["path"]),
                command=gpt_conf.get("command"),
                isolation_guard=self.isolation_guard,
                git_manager=self.git_manager,
            )

    def get_runner_for_proposal(self, proposal: Proposal) -> AgentRunner:
        """Determines the appropriate agent runner for the proposal."""
        assigned_branch = proposal.metadata.assigned_branch
        author = proposal.metadata.author_model.value

        # Map branch name to workspace key
        branch_map = {
            "dev": "gemini",
            "claude": "claude",
            "chatgpt": "chatgpt",
        }

        target_key = branch_map.get(assigned_branch, author)
        if target_key in self.runners:
            return self.runners[target_key]
        raise ValueError(f"No configured runner for target '{target_key}' (branch: '{assigned_branch}')")

    def build_implementation_prompt(self, proposal: Proposal) -> str:
        """Constructs an executable prompt for the agent to implement the proposal."""
        mod_files = "\n".join(f"- {f}" for f in proposal.proposed_files) if proposal.proposed_files else "See proposal body."
        prompt = (
            f"Please implement algorithmic proposal {proposal.metadata.id}: {proposal.metadata.title}\n\n"
            f"Target Strategy: {proposal.metadata.target_strategy.value}\n"
            f"Hypothesis:\n{proposal.hypothesis}\n\n"
            f"Target Files:\n{mod_files}\n\n"
            f"Instructions:\n"
            f"1. Implement the requested algorithmic modifications.\n"
            f"2. DO NOT modify any evaluation core files (evaluation/folds.py, evaluation/runner.py, evaluation/metrics.py).\n"
            f"3. Run quick smoke tests to ensure code executes without syntax or import errors.\n"
            f"4. Commit your changes with commit message 'feat({proposal.metadata.target_strategy.value}): {proposal.metadata.id} - {proposal.metadata.title}'."
        )
        return prompt

    def run_proposal_execution(
        self,
        proposal: Proposal,
        on_output: Optional[Callable[[str], None]] = None,
        auto_eval: bool = True,
    ) -> tuple[AgentExecutionResult, Optional[ScorecardSummary]]:
        """
        Full in-process execution pipeline:
        1. Transition state: APPROVED -> IMPLEMENTING
        2. Run agent in workspace with streaming output
        3. Detect commit
        4. If committed and auto_eval=True:
           - Transition state: IMPLEMENTING -> AB_TESTING
           - Run evaluation.cli via EvalBridge
           - Transition state: AB_TESTING -> EVALUATED
           - Attach scorecard to proposal
        """
        # 1. State transition
        if proposal.metadata.status != ProposalStatus.APPROVED:
            raise ValueError(f"Proposal must be APPROVED before execution (current: {proposal.metadata.status.value})")

        ProposalStateMachine.transition(proposal, ProposalStatus.IMPLEMENTING)
        if self.proposal_manager:
            self.proposal_manager.save_proposal(proposal)

        runner = self.get_runner_for_proposal(proposal)
        prompt = self.build_implementation_prompt(proposal)

        # 2. Run agent
        if on_output:
            on_output(f"[*] Spawning agent '{runner.name}' in workspace '{runner.workspace_path}'...")

        result = runner.execute(prompt=prompt, on_output=on_output)

        if not result.success:
            if on_output:
                on_output(f"[!] Agent execution failed: {result.error}")
            return result, None

        if not result.commit_sha:
            if on_output:
                on_output(f"[!] Agent finished without producing a new git commit.")
            return result, None

        if on_output:
            on_output(f"[+] Agent successfully committed changes: {result.commit_sha[:8]}")

        # 3. Trigger evaluation if enabled
        scorecard = None
        if auto_eval and self.eval_bridge:
            ProposalStateMachine.transition(proposal, ProposalStatus.AB_TESTING)
            if self.proposal_manager:
                self.proposal_manager.save_proposal(proposal)

            eval_mode = proposal.metadata.acceptance_criteria.eval_mode
            target_strat = proposal.metadata.target_strategy.value
            variant_name = proposal.metadata.id.lower()

            if on_output:
                on_output(f"[*] Triggering automated evaluation: --strategy {target_strat} --mode {eval_mode} --ab {variant_name}...")

            scorecard = self.eval_bridge.run_evaluation(
                workspace_path=runner.workspace_path,
                strategy=target_strat,
                mode=eval_mode,
                ab_variant=variant_name,
                stdout_callback=on_output,
            )

            if scorecard:
                proposal.evaluation_results = scorecard
                ProposalStateMachine.transition(proposal, ProposalStatus.EVALUATED)
                if self.proposal_manager:
                    self.proposal_manager.save_proposal(proposal)
                if on_output:
                    on_output(f"[+] Evaluation finished: Top-1 Delta = {scorecard.delta_top1_precision:+.2f}%, Total PnL Delta = {scorecard.delta_total_pnl:+.2f}%")

        return result, scorecard
