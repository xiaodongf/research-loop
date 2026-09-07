"""Stage 2: Cross-Branch Execution & Multi-Workspace Sync Tests (test_cross_execution.py).

Verifies:
1. Cross-branch dispatch (Model A proposal assigned to Model B branch).
2. Promotion to golden main branch & RFC relocation to proposals/promoted/.
3. Multi-workspace rebase synchronization across dev, claude, and chatgpt branches.
4. Rebase conflict guard and automatic clean rollback (git rebase --abort).
"""
import subprocess
from pathlib import Path
import pytest

from src.core.schema import ProposalStatus, ModelName, StrategyTarget, ScorecardSummary
from src.core.state_machine import ProposalStateMachine
from src.core.proposal_manager import ProposalManager
from src.core.git_manager import GitManager, RebaseConflictError
from src.agents.dispatcher import AgentDispatcher


@pytest.fixture
def multi_workspace_setup(tmp_path):
    """
    Creates a golden upstream Git repo with 'main', and clones 3 separate workspaces:
    - trading (dev)
    - trading-claude (claude)
    - trading-gpt (chatgpt)
    """
    upstream = tmp_path / "golden_repo"
    upstream.mkdir()
    subprocess.run(["git", "init", "--bare", "-b", "main"], cwd=str(upstream), check=True, stdout=subprocess.DEVNULL)

    # Initial seed clone to establish initial main commit
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(upstream), str(seed)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Seed"], cwd=str(seed), check=True)
    subprocess.run(["git", "config", "user.email", "seed@trading.internal"], cwd=str(seed), check=True)
    (seed / "README.md").write_text("# Trading System Golden Baseline\n")
    (seed / "features.py").write_text("def base_feature(): return 1\n")
    subprocess.run(["git", "add", "."], cwd=str(seed), check=True)
    subprocess.run(["git", "commit", "-m", "chore: initial golden baseline"], cwd=str(seed), check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=str(seed), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clone the 3 model workspaces from upstream
    workspaces = {}
    branch_map = {
        "gemini": ("trading", "dev"),
        "claude": ("trading-claude", "claude"),
        "chatgpt": ("trading-gpt", "chatgpt"),
    }

    for key, (folder, branch) in branch_map.items():
        ws_dir = tmp_path / folder
        subprocess.run(["git", "clone", str(upstream), str(ws_dir)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.name", f"{key.title()} Agent"], cwd=str(ws_dir), check=True)
        subprocess.run(["git", "config", "user.email", f"{key}@trading.internal"], cwd=str(ws_dir), check=True)
        # Checkout dedicated branch
        subprocess.run(["git", "checkout", "-b", branch], cwd=str(ws_dir), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        workspaces[key] = ws_dir

    # Also keep a local working copy of golden main
    golden_ws = tmp_path / "golden_ws"
    subprocess.run(["git", "clone", str(upstream), str(golden_ws)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Harness Admin"], cwd=str(golden_ws), check=True)
    subprocess.run(["git", "config", "user.email", "admin@trading.internal"], cwd=str(golden_ws), check=True)

    return {
        "upstream": upstream,
        "golden_ws": golden_ws,
        "workspaces": workspaces,
    }


def test_cross_branch_dispatch(multi_workspace_setup):
    """Proposal authored by Gemini is assigned for implementation on Claude branch."""
    workspaces = multi_workspace_setup["workspaces"]
    claude_ws = workspaces["claude"]

    config = {
        "workspaces": {
            "gemini": {"path": str(workspaces["gemini"])},
            "claude": {
                "path": str(claude_ws),
                "binary": (
                    "mock:echo 'def claude_feature(): return 42' >> features.py && "
                    "git add features.py && "
                    "git commit -m 'feat(standalone_tb): PROP-005 - Implemented by Claude'"
                ),
            },
            "chatgpt": {"path": str(workspaces["chatgpt"])},
        }
    }

    pm = ProposalManager(base_dir=multi_workspace_setup["golden_ws"] / "proposals")
    proposal = pm.create_proposal(
        title="Cross Model Implementation",
        author_model=ModelName.GEMINI,
        target_strategy=StrategyTarget.STANDALONE_TB,
        assigned_branch="claude",
        hypothesis="Gemini synthesized idea executed by Claude Code.",
        proposed_files=["features.py"],
    )
    ProposalStateMachine.submit_to_human(proposal)
    ProposalStateMachine.approve(proposal, assigned_branch="claude")

    dispatcher = AgentDispatcher(config=config, proposal_manager=pm)
    runner = dispatcher.get_runner_for_proposal(proposal)

    # Asserts Claude runner was selected even though author was Gemini
    assert runner.name == "claude"
    assert runner.workspace_path == claude_ws

    result, _ = dispatcher.run_proposal_execution(proposal, auto_eval=False)
    assert result.success is True
    assert result.commit_sha is not None
    assert "Implemented by Claude" in result.output or (claude_ws / "features.py").read_text().count("claude_feature") > 0


def test_promotion_to_main(multi_workspace_setup):
    """Winning proposal commit is merged into golden main and RFC relocated to proposals/promoted/."""
    workspaces = multi_workspace_setup["workspaces"]
    golden_ws = multi_workspace_setup["golden_ws"]
    claude_ws = workspaces["claude"]

    git_mgr = GitManager(workspaces=workspaces, golden_branch="main")

    # Claude makes winning commit
    (claude_ws / "features.py").write_text("def winning_feature(): return 100\n")
    subprocess.run(["git", "add", "features.py"], cwd=str(claude_ws), check=True)
    subprocess.run(["git", "commit", "-m", "feat(standalone_tb): PROP-010 - Winning Feature"], cwd=str(claude_ws), check=True)
    winning_sha = git_mgr.get_head_sha(claude_ws)

    # Promote to golden main
    new_main_sha = git_mgr.promote_to_main(
        candidate_ws=claude_ws,
        golden_ws=golden_ws,
        proposal_id="PROP-010",
        title="Winning Feature",
        commit_sha=winning_sha,
    )

    assert new_main_sha != winning_sha  # Merge commit created
    commit_log = subprocess.check_output(["git", "-C", str(golden_ws), "log", "-1", "--pretty=%s"]).decode()
    assert "PROP-010 - Winning Feature" in commit_log

    # Proposal relocation test
    pm = ProposalManager(base_dir=golden_ws / "proposals")
    proposal = pm.create_proposal(
        title="Winning Feature",
        author_model=ModelName.CLAUDE,
        target_strategy=StrategyTarget.STANDALONE_TB,
        assigned_branch="claude",
        hypothesis="Winning delta",
    )
    ProposalStateMachine.submit_to_human(proposal)
    ProposalStateMachine.approve(proposal)
    ProposalStateMachine.transition(proposal, ProposalStatus.IMPLEMENTING)
    ProposalStateMachine.transition(proposal, ProposalStatus.AB_TESTING)
    ProposalStateMachine.transition(proposal, ProposalStatus.EVALUATED)

    promoted_path = pm.promote_proposal(proposal)
    assert promoted_path.exists()
    assert "promoted" in str(promoted_path)
    assert not (pm.active_dir / f"{proposal.id}.md").exists()


def test_multi_workspace_rebase_sync(multi_workspace_setup):
    """
    Verifies that when golden main receives an update:
    All 3 model workspaces (dev, claude, chatgpt) are synchronized to golden main.
    """
    workspaces = multi_workspace_setup["workspaces"]
    golden_ws = multi_workspace_setup["golden_ws"]
    upstream = multi_workspace_setup["upstream"]
    git_mgr = GitManager(workspaces=workspaces, golden_branch="main")

    # Add a new commit to golden_ws main and push upstream
    (golden_ws / "new_core_module.py").write_text("NEW_MODULE = True\n")
    subprocess.run(["git", "add", "."], cwd=str(golden_ws), check=True)
    subprocess.run(["git", "commit", "-m", "feat(core): new promoted foundation"], cwd=str(golden_ws), check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=str(golden_ws), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    new_main_sha = git_mgr.get_head_sha(golden_ws)

    # Sync all workspaces from golden main
    results = git_mgr.sync_all_workspaces(golden_ws=golden_ws)
    assert all(results.values()) is True

    # Assert each workspace now has new_core_module.py and the commit in its log
    for key, ws_path in workspaces.items():
        assert (ws_path / "new_core_module.py").exists(), f"{key} missing new_core_module.py after sync"
        log = subprocess.check_output(["git", "-C", str(ws_path), "log", "-n", "3", "--pretty=%s"]).decode()
        assert "feat(core): new promoted foundation" in log, f"{key} missing commit log"


def test_rebase_conflict_guard_and_rollback(multi_workspace_setup):
    """
    Simulates a conflicting change on one branch during rebase.
    Verifies that:
    1. RebaseConflictError is raised.
    2. The rebase is cleanly aborted (git rebase --abort).
    3. The workspace is preserved without dangling conflict markers or lock files.
    """
    workspaces = multi_workspace_setup["workspaces"]
    golden_ws = multi_workspace_setup["golden_ws"]
    chatgpt_ws = workspaces["chatgpt"]
    git_mgr = GitManager(workspaces=workspaces, golden_branch="main")

    # 1. Main updates features.py line 1 to something new
    (golden_ws / "features.py").write_text("def base_feature(): return 'GOLDEN_VERSION'\n")
    subprocess.run(["git", "add", "features.py"], cwd=str(golden_ws), check=True)
    subprocess.run(["git", "commit", "-m", "feat(main): change base_feature to golden"], cwd=str(golden_ws), check=True)

    # 2. chatgpt updates features.py line 1 to a conflicting value locally
    (chatgpt_ws / "features.py").write_text("def base_feature(): return 'CHATGPT_CONFLICT_VERSION'\n")
    subprocess.run(["git", "add", "features.py"], cwd=str(chatgpt_ws), check=True)
    subprocess.run(["git", "commit", "-m", "feat(chatgpt): conflicting edit"], cwd=str(chatgpt_ws), check=True)
    pre_rebase_sha = git_mgr.get_head_sha(chatgpt_ws)

    # 3. Attempting to rebase chatgpt onto golden main should detect conflict, abort cleanly, and raise RebaseConflictError
    subprocess.run(["git", "fetch", str(golden_ws), "main"], cwd=str(chatgpt_ws), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with pytest.raises(RebaseConflictError) as exc_info:
        git_mgr.rebase_branch_with_rollback(chatgpt_ws, "FETCH_HEAD")

    assert "Rebase conflict detected" in str(exc_info.value)
    assert "Rebase was aborted cleanly" in str(exc_info.value)

    # 4. Verify working tree is clean and restored to pre-rebase commit
    assert git_mgr.is_clean(chatgpt_ws) is True
    assert git_mgr.get_head_sha(chatgpt_ws) == pre_rebase_sha
    assert (chatgpt_ws / "features.py").read_text() == "def base_feature(): return 'CHATGPT_CONFLICT_VERSION'\n"
    # Ensure no .git/rebase-apply or .git/rebase-merge exists
    assert not (chatgpt_ws / ".git" / "rebase-apply").exists()
    assert not (chatgpt_ws / ".git" / "rebase-merge").exists()
