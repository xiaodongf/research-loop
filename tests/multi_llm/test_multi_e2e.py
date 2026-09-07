"""Stage 2: Multi-LLM End-to-End Collaborative Loop (test_multi_e2e.py).

Simulates the complete 3-model collaborative research cycle:
1. Gemini drafts algorithmic proposal PROP-MULTI-001.
2. Routed to Claude for adversarial peer review -> Claude appends cautionary critique.
3. Human reviews multi-model consensus and approves implementation on Claude branch.
4. Claude in-process agent executes code edits and commits.
5. Harness runs automatic evaluation bridge and parses A/B scorecard.
6. Winning commit is promoted into golden main branch.
7. Harness synchronizes the other branches (dev and chatgpt) to the new main baseline.
"""
import subprocess
from pathlib import Path
import pytest

from src.core.schema import ProposalStatus, ModelName, StrategyTarget, ReviewVerdict
from src.core.state_machine import ProposalStateMachine
from src.core.proposal_manager import ProposalManager
from src.core.git_manager import GitManager
from src.agents.dispatcher import AgentDispatcher
from src.harness.eval_bridge import EvalBridge
from src.harness.csv_parser import CSVParser


@pytest.fixture
def multi_e2e_environment(tmp_path):
    """Sets up golden repository and 3 model workspaces."""
    upstream = tmp_path / "golden_repo"
    upstream.mkdir()
    subprocess.run(["git", "init", "--bare", "-b", "main"], cwd=str(upstream), check=True, stdout=subprocess.DEVNULL)

    # Initial main commit
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(upstream), str(seed)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Initial Seed"], cwd=str(seed), check=True)
    subprocess.run(["git", "config", "user.email", "seed@trading.internal"], cwd=str(seed), check=True)
    (seed / ".gitignore").write_text("*.csv\nevaluation/outputs/\n__pycache__/\n")
    (seed / "features.py").write_text("def base_feature(): return 1.0\n")
    subprocess.run(["git", "add", "."], cwd=str(seed), check=True)
    subprocess.run(["git", "commit", "-m", "chore: initial main baseline"], cwd=str(seed), check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=str(seed), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Workspaces
    workspaces = {}
    branch_map = {
        "gemini": ("trading", "dev"),
        "claude": ("trading-claude", "claude"),
        "chatgpt": ("trading-gpt", "chatgpt"),
    }
    for key, (folder, branch) in branch_map.items():
        ws = tmp_path / folder
        subprocess.run(["git", "clone", str(upstream), str(ws)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.name", f"{key.title()} Agent"], cwd=str(ws), check=True)
        subprocess.run(["git", "config", "user.email", f"{key}@trading.internal"], cwd=str(ws), check=True)
        subprocess.run(["git", "checkout", "-b", branch], cwd=str(ws), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        workspaces[key] = ws

    golden_ws = tmp_path / "golden_ws"
    subprocess.run(["git", "clone", str(upstream), str(golden_ws)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Admin"], cwd=str(golden_ws), check=True)
    subprocess.run(["git", "config", "user.email", "admin@trading.internal"], cwd=str(golden_ws), check=True)

    return {
        "workspaces": workspaces,
        "golden_ws": golden_ws,
    }


def test_full_multi_llm_collaborative_loop(tmp_path, multi_e2e_environment):
    workspaces = multi_e2e_environment["workspaces"]
    golden_ws = multi_e2e_environment["golden_ws"]
    claude_ws = workspaces["claude"]

    git_mgr = GitManager(workspaces=workspaces, golden_branch="main")
    pm = ProposalManager(base_dir=tmp_path / "proposals")

    # Step 1: Gemini drafts Proposal PROP-MULTI-001
    proposal = pm.create_proposal(
        title="Multi-Horizon Volatility Estimator",
        author_model=ModelName.GEMINI,
        target_strategy=StrategyTarget.STANDALONE_TB,
        assigned_branch="claude",
        hypothesis="Multi-horizon Garman-Klass volatility improves Sharpe ratio in volatile regimes.",
        proposed_files=["features.py"],
        implementation_spec="Calculate 5d and 20d Garman-Klass volatility with shift(1).",
    )
    assert proposal.metadata.status == ProposalStatus.DRAFT

    # Step 2: Human routes to Claude for peer review
    ProposalStateMachine.submit_to_human(proposal)
    assert proposal.metadata.status == ProposalStatus.HUMAN_REVIEW

    ProposalStateMachine.route_to_peer(proposal, reviewer=ModelName.CLAUDE)
    assert proposal.metadata.status == ProposalStatus.PEER_REVIEW

    # Claude records critique with caution
    ProposalStateMachine.add_peer_critique(
        proposal=proposal,
        reviewer=ModelName.CLAUDE,
        verdict=ReviewVerdict.APPROVED_WITH_CAUTION,
        comment="Mathematical formulation verified. Ensure unshifted current day bar is not included.",
    )
    # Returns to HUMAN_REVIEW
    assert proposal.metadata.status == ProposalStatus.HUMAN_REVIEW
    assert len(proposal.metadata.reviewers) == 1
    pm.save_proposal(proposal)

    # Step 3: Human reviews debate and approves for execution on Claude branch
    ProposalStateMachine.approve(proposal, assigned_branch="claude")
    assert proposal.metadata.status == ProposalStatus.APPROVED
    pm.save_proposal(proposal)

    # Step 4: Dispatcher spawns Claude Code in workspace/trading-claude
    results_dir = claude_ws / "evaluation" / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    mock_csv_content = (
        "fold,regime,spy,base_n,base_win,base_total_return,base_ret_per_trade,base_max_dd,base_avg_exp,base_max_exp,base_sharpe,base_prec1,var_n,var_win,var_total_return,var_ret_per_trade,var_max_dd,var_avg_exp,var_max_exp,var_sharpe,var_prec1\n"
        "2012-07-01,Bull-Lo,3.4,30,0.60,10.0,3.3,-5.0,50.0,100.0,0.5,40.0,30,0.70,16.0,5.3,-3.8,50.0,100.0,0.65,48.0\n"
        "2020-09-01,Bear-Hi,-6.0,40,0.50,5.0,1.25,-6.0,40.0,80.0,0.3,30.0,40,0.62,11.0,2.75,-4.0,40.0,80.0,0.45,38.0\n"
    )

    eval_bridge = EvalBridge(python_bin=Path("python3"), results_dir=results_dir)
    def mock_eval_run(workspace_path, strategy="standalone_tb", mode="quick", ab_variant="aa", timeout=None, stdout_callback=None):
        csv_file = results_dir / f"{strategy}_ab-{ab_variant}_QUICK_20260906_120000_mock.csv"
        csv_file.write_text(mock_csv_content)
        return CSVParser.parse_result_file(csv_file)
    eval_bridge.run_evaluation = mock_eval_run

    config = {
        "workspaces": {
            "gemini": {"path": str(workspaces["gemini"])},
            "claude": {
                "path": str(claude_ws),
                "binary": (
                    "mock:echo 'def garman_klass_vol(): return 0.015' >> features.py && "
                    "git add features.py && "
                    "git commit -m 'feat(standalone_tb): PROP-001 - Multi-Horizon Volatility Estimator'"
                ),
            },
            "chatgpt": {"path": str(workspaces["chatgpt"])},
        }
    }

    dispatcher = AgentDispatcher(config=config, proposal_manager=pm, eval_bridge=eval_bridge)
    exec_res, scorecard = dispatcher.run_proposal_execution(proposal, auto_eval=True)

    # Step 5: Assertions on execution & backtest scorecard
    assert exec_res.success is True
    assert exec_res.commit_sha is not None
    assert proposal.metadata.status == ProposalStatus.EVALUATED
    assert scorecard is not None
    assert scorecard.delta_total_pnl == pytest.approx(12.0)  # (16 + 11) - (10 + 5) = 27 - 15 = 12.0
    assert scorecard.delta_top1_precision == pytest.approx(8.0)  # avg(48, 38) - avg(40, 30) = 43 - 35 = 8.0

    # Step 6: Promote winning candidate to golden main branch
    new_main_sha = git_mgr.promote_to_main(
        candidate_ws=claude_ws,
        golden_ws=golden_ws,
        proposal_id=proposal.id,
        title=proposal.metadata.title,
    )
    assert new_main_sha is not None

    promoted_file = pm.promote_proposal(proposal)
    assert promoted_file.exists()
    assert "promoted" in str(promoted_file)

    # Step 7: Synchronize all other workspaces (dev and chatgpt) to new main baseline
    sync_results = git_mgr.sync_all_workspaces(golden_ws=golden_ws, skip_key="claude")
    assert all(sync_results.values()) is True

    for key in ("gemini", "chatgpt"):
        ws = workspaces[key]
        assert (ws / "features.py").read_text().count("garman_klass_vol") > 0
        git_log = subprocess.check_output(["git", "-C", str(ws), "log", "-n", "3", "--pretty=%s"]).decode()
        assert f"{proposal.id} - {proposal.metadata.title}" in git_log
