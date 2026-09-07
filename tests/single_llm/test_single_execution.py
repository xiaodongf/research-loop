"""Stage 1: Single-LLM Execution and Backtest Tests.

Verifies:
1. In-process agent execution inside isolated workspace.
2. Real-time stdout/stderr streaming callback.
3. Automatic Git commit watcher detection.
4. Automatic triggering of evaluation.cli.
5. Ingestion of results CSV and Delta scorecard computation.
"""
import subprocess, time
from pathlib import Path
import pytest

from src.core.schema import Proposal, ProposalStatus, ModelName, StrategyTarget, AcceptanceCriteria
from src.core.state_machine import ProposalStateMachine
from src.core.proposal_manager import ProposalManager
from src.core.git_manager import GitManager
from src.core.isolation_guard import WorkspaceIsolationGuard
from src.agents.claude_runner import ClaudeCodeRunner
from src.agents.dispatcher import AgentDispatcher
from src.harness.eval_bridge import EvalBridge


@pytest.fixture
def mock_single_workspace(tmp_path):
    """Sets up a mock Git workspace for single model testing."""
    ws = tmp_path / "trading-claude"
    ws.mkdir()
    # Init git repo
    subprocess.run(["git", "init"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Test Agent"], cwd=str(ws), check=True)
    subprocess.run(["git", "config", "user.email", "agent@test.com"], cwd=str(ws), check=True)

    # Initial file and commit
    dummy_feat = ws / "feature.py"
    dummy_feat.write_text("def my_feature(): return 1\n")
    subprocess.run(["git", "add", "."], cwd=str(ws), check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(ws), check=True)

    # Output dir for mock evaluations
    eval_out = ws / "evaluation" / "outputs" / "results"
    eval_out.mkdir(parents=True)

    return ws


def test_single_in_process_runner(mock_single_workspace):
    """Test spawning agent process inside workspace and verifying cwd isolation."""
    runner = ClaudeCodeRunner(
        workspace_path=mock_single_workspace,
        binary_path="mock:pwd",
    )
    result = runner.execute("Test prompt")
    assert result.success is True
    assert result.exit_code == 0
    assert str(mock_single_workspace.resolve()) in result.output


def test_single_stdout_streaming(mock_single_workspace):
    """Test real-time stdout streaming callback."""
    runner = ClaudeCodeRunner(
        workspace_path=mock_single_workspace,
        binary_path="mock:echo 'line 1' && echo 'line 2' && echo 'line 3'",
    )
    streamed_lines = []

    def on_output(line: str):
        streamed_lines.append(line)

    result = runner.execute("Test prompt", on_output=on_output)
    assert result.success is True
    assert "line 1" in streamed_lines
    assert "line 2" in streamed_lines
    assert "line 3" in streamed_lines


def test_single_git_commit_detection(mock_single_workspace):
    """Test agent runner detecting new commit created by agent."""
    git_mgr = GitManager({"claude": mock_single_workspace})
    initial_sha = git_mgr.get_head_sha(mock_single_workspace)

    # Simulate an agent making a commit
    agent_cmd = (
        "mock:echo 'def updated_feature(): return 42' > feature.py && "
        "git add feature.py && "
        "git commit -m 'feat(model): add updated feature'"
    )
    runner = ClaudeCodeRunner(
        workspace_path=mock_single_workspace,
        binary_path=agent_cmd,
        git_manager=git_mgr,
    )

    result = runner.execute("Implement update")
    assert result.success is True
    assert result.commit_sha is not None
    assert result.commit_sha != initial_sha

    new_sha = git_mgr.get_head_sha(mock_single_workspace)
    assert result.commit_sha == new_sha
    assert "add updated feature" in git_mgr.get_commit_message(mock_single_workspace)


def test_single_auto_trigger_evaluation_and_scorecard(tmp_path, mock_single_workspace):
    """Test end-to-end execution: approved proposal -> agent commit -> auto eval -> scorecard attached."""
    proposals_dir = tmp_path / "proposals"
    pm = ProposalManager(base_dir=proposals_dir)

    # Create mock CSV output matching evaluation.cli format
    results_dir = mock_single_workspace / "evaluation" / "outputs" / "results"
    mock_csv_content = (
        "fold,regime,spy,base_n,base_win,base_total_return,base_ret_per_trade,base_max_dd,base_avg_exp,base_max_exp,base_sharpe,base_prec1,var_n,var_win,var_total_return,var_ret_per_trade,var_max_dd,var_avg_exp,var_max_exp,var_sharpe,var_prec1\n"
        "2012-07-01,Bull-Lo,3.4,30,0.60,10.0,3.3,-5.0,50.0,100.0,0.5,40.0,30,0.70,15.0,5.0,-4.0,50.0,100.0,0.6,50.0\n"
        "2020-09-01,Bear-Hi,-6.0,40,0.50,5.0,1.25,-6.0,40.0,80.0,0.3,50.0,40,0.60,10.0,2.5,-4.0,40.0,80.0,0.4,53.0\n"
    )

    # Setup eval bridge
    eval_bridge = EvalBridge(
        python_bin=Path("python3"),
        results_dir=results_dir,
    )

    # Create and approve proposal via human review gate
    proposal = pm.create_proposal(
        title="Dynamic Volatility Scaling",
        author_model=ModelName.CLAUDE,
        target_strategy=StrategyTarget.STANDALONE_TB,
        assigned_branch="claude",
        hypothesis="Testing auto eval trigger",
        proposed_files=["feature.py"],
    )
    ProposalStateMachine.transition(proposal, ProposalStatus.HUMAN_REVIEW)
    ProposalStateMachine.transition(proposal, ProposalStatus.APPROVED)
    pm.save_proposal(proposal)

    # Config for dispatcher
    config = {
        "workspaces": {
            "claude": {
                "name": "Claude Code",
                "path": str(mock_single_workspace),
                "binary": (
                    "mock:echo 'x = 100' >> feature.py && "
                    "git add feature.py && "
                    "git commit -m 'feat(standalone_tb): PROP-001 - Dynamic Volatility Scaling'"
                ),
            }
        }
    }

    dispatcher = AgentDispatcher(
        config=config,
        proposal_manager=pm,
        eval_bridge=eval_bridge,
    )

    # Patch eval bridge run to write the mock CSV and return parsed scorecard
    from src.harness.csv_parser import CSVParser
    def custom_run_eval(workspace_path, strategy="standalone_tb", mode="quick", ab_variant="aa", timeout=None, stdout_callback=None):
        csv_file = results_dir / f"{strategy}_ab-{ab_variant}_QUICK_20260906_120000_mock.csv"
        csv_file.write_text(mock_csv_content)
        if stdout_callback:
            stdout_callback(f"[Harness] Found evaluation output CSV: {csv_file.name}")
        return CSVParser.parse_result_file(csv_file)
    eval_bridge.run_evaluation = custom_run_eval

    logs = []
    exec_result, scorecard = dispatcher.run_proposal_execution(
        proposal=proposal,
        on_output=lambda l: logs.append(l),
        auto_eval=True,
    )

    assert exec_result.success is True
    assert exec_result.commit_sha is not None
    assert proposal.metadata.status == ProposalStatus.EVALUATED
    assert scorecard is not None
    assert scorecard.base_top1_precision == 45.0
    assert scorecard.variant_top1_precision == 51.5
    assert scorecard.delta_top1_precision == pytest.approx(6.5)
    assert scorecard.delta_total_pnl == pytest.approx(10.0)
