"""Stage 1: Single-LLM End-to-End Sanity Loop (test_single_e2e.py).

Tests the complete single-model cycle:
1. Ideation prompt generation & RFC creation (PROP-SINGLE-001).
2. Human-in-the-loop review and approval.
3. In-process agent execution in isolated workspace with commit creation.
4. Automatic evaluation bridge trigger.
5. Scorecard ingestion and proposal state transition to EVALUATED.
"""
import subprocess
from pathlib import Path
import pytest

from src.core.schema import ProposalStatus, ModelName, StrategyTarget
from src.core.state_machine import ProposalStateMachine
from src.core.proposal_manager import ProposalManager
from src.core.ideation import ContextBuilder, PromptBuilder
from src.agents.dispatcher import AgentDispatcher
from src.harness.eval_bridge import EvalBridge
from src.harness.csv_parser import CSVParser


@pytest.fixture
def e2e_single_workspace(tmp_path):
    ws = tmp_path / "trading-dev"
    ws.mkdir()
    subprocess.run(["git", "init"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Gemini Agent"], cwd=str(ws), check=True)
    subprocess.run(["git", "config", "user.email", "gemini@trading.internal"], cwd=str(ws), check=True)

    # Initial code
    (ws / "strategy.py").write_text("class Strategy:\n    def run(self): return 1.0\n")
    subprocess.run(["git", "add", "."], cwd=str(ws), check=True)
    subprocess.run(["git", "commit", "-m", "chore: initial baseline"], cwd=str(ws), check=True)

    results_dir = ws / "evaluation" / "outputs" / "results"
    results_dir.mkdir(parents=True)
    return ws


def test_single_model_e2e_loop(tmp_path, e2e_single_workspace):
    proposals_dir = tmp_path / "proposals"
    pm = ProposalManager(base_dir=proposals_dir)

    # 1. Ideation context building
    context = ContextBuilder.build(
        strategy=StrategyTarget.STANDALONE_TB,
        results_dir=e2e_single_workspace / "evaluation" / "outputs" / "results",
        rejected_dir=proposals_dir / "rejected",
    )
    prompt = PromptBuilder.create_ideation_prompt(
        author_model=ModelName.GEMINI,
        strategy=StrategyTarget.STANDALONE_TB,
        theme="volatility_normalization",
        context=context,
        next_id="PROP-001",
    )
    assert "PROP-001" in prompt
    assert "author_model: gemini" in prompt

    # 2. RFC Creation
    proposal = pm.create_proposal(
        title="Adaptive Volatility Window on Standalone TB",
        author_model=ModelName.GEMINI,
        target_strategy=StrategyTarget.STANDALONE_TB,
        assigned_branch="dev",
        hypothesis="Dynamic volatility scaling stabilizes quantile ranking during high-vol regimes.",
        proposed_files=["strategy.py"],
        implementation_spec="Implement 20-day Parkinson rolling volatility calculation.",
    )
    assert proposal.metadata.status == ProposalStatus.DRAFT

    # 3. Submit to Human Review Gate
    ProposalStateMachine.submit_to_human(proposal)
    assert proposal.metadata.status == ProposalStatus.HUMAN_REVIEW

    # 4. Human Approval
    ProposalStateMachine.approve(proposal, assigned_branch="dev")
    assert proposal.metadata.status == ProposalStatus.APPROVED
    pm.save_proposal(proposal)

    # 5. Agent Dispatch & In-Process Execution
    results_dir = e2e_single_workspace / "evaluation" / "outputs" / "results"
    mock_csv_content = (
        "fold,regime,spy,base_n,base_win,base_total_return,base_ret_per_trade,base_max_dd,base_avg_exp,base_max_exp,base_sharpe,base_prec1,var_n,var_win,var_total_return,var_ret_per_trade,var_max_dd,var_avg_exp,var_max_exp,var_sharpe,var_prec1\n"
        "2012-07-01,Bull-Lo,3.4,30,0.60,10.0,3.3,-5.0,50.0,100.0,0.5,40.0,30,0.65,12.0,4.0,-4.0,50.0,100.0,0.55,44.0\n"
        "2020-09-01,Bear-Hi,-6.0,40,0.50,5.0,1.25,-6.0,40.0,80.0,0.3,30.0,40,0.58,9.0,2.25,-4.5,40.0,80.0,0.42,36.0\n"
    )

    eval_bridge = EvalBridge(
        python_bin=Path("python3"),
        results_dir=results_dir,
    )

    def mock_eval_run(workspace_path, strategy="standalone_tb", mode="quick", ab_variant="aa", timeout=None, stdout_callback=None):
        csv_file = results_dir / f"{strategy}_ab-{ab_variant}_QUICK_20260906_120000_mock.csv"
        csv_file.write_text(mock_csv_content)
        return CSVParser.parse_result_file(csv_file)
    eval_bridge.run_evaluation = mock_eval_run

    config = {
        "workspaces": {
            "gemini": {
                "name": "Gemini (Antigravity)",
                "path": str(e2e_single_workspace),
                "command": (
                    "echo 'def parkinson_vol(): return 0.02' >> strategy.py && "
                    "git add strategy.py && "
                    "git commit -m 'feat(standalone_tb): PROP-001 - Adaptive Volatility Window'"
                ),
            }
        }
    }

    dispatcher = AgentDispatcher(
        config=config,
        proposal_manager=pm,
        eval_bridge=eval_bridge,
    )

    log_stream = []
    result, scorecard = dispatcher.run_proposal_execution(
        proposal=proposal,
        on_output=lambda l: log_stream.append(l),
        auto_eval=True,
    )

    # 6. Verifications
    assert result.success is True
    assert result.commit_sha is not None
    assert proposal.metadata.status == ProposalStatus.EVALUATED
    assert scorecard is not None

    # Base vs Variant assertions
    assert scorecard.base_pnl_total == 15.0  # 10 + 5
    assert scorecard.var_pnl_total == 21.0   # 12 + 9
    assert scorecard.delta_total_pnl == pytest.approx(6.0)
    assert scorecard.delta_top1_precision == pytest.approx(5.0)  # avg(44, 36) - avg(40, 30) = 40 - 35 = 5
    assert scorecard.fold_wins == 2

    # Verify persisted proposal markdown contains updated status and scorecard
    saved_proposal = pm.load_proposal(proposal.id)
    assert saved_proposal.metadata.status == ProposalStatus.EVALUATED
    assert saved_proposal.scorecard_summary is not None
