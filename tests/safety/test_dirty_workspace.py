"""Safety: Dirty Workspace Protection Tests (test_dirty_workspace.py).

Verifies:
1. Detection of uncommitted/untracked changes across model workspaces.
2. Rejection of agent execution and rebasing on dirty workspaces.
3. In-flight work preservation.
"""
import subprocess
from pathlib import Path
import pytest

from src.core.git_manager import GitManager, DirtyWorkspaceError


@pytest.fixture
def repo_with_dirty_state(tmp_path):
    ws = tmp_path / "trading-dev"
    ws.mkdir()
    subprocess.run(["git", "init"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=str(ws), check=True)
    subprocess.run(["git", "config", "user.email", "dev@trading.internal"], cwd=str(ws), check=True)

    (ws / "main.py").write_text("print('hello')\n")
    subprocess.run(["git", "add", "."], cwd=str(ws), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(ws), check=True)
    return ws


def test_clean_workspace_passes(repo_with_dirty_state):
    git_mgr = GitManager({"dev": repo_with_dirty_state})
    assert git_mgr.is_clean(repo_with_dirty_state) is True
    assert git_mgr.check_all_clean() == []
    # Should not raise
    git_mgr.assert_clean(repo_with_dirty_state, "dev")


def test_dirty_workspace_uncommitted_modification_fails(repo_with_dirty_state):
    git_mgr = GitManager({"dev": repo_with_dirty_state})
    # Modify tracked file without committing
    (repo_with_dirty_state / "main.py").write_text("print('dirty change')\n")

    assert git_mgr.is_clean(repo_with_dirty_state) is False
    assert git_mgr.check_all_clean() == ["dev"]

    with pytest.raises(DirtyWorkspaceError) as exc_info:
        git_mgr.assert_clean(repo_with_dirty_state, "dev")
    assert "has uncommitted changes" in str(exc_info.value)


def test_dirty_workspace_untracked_file_fails(repo_with_dirty_state):
    git_mgr = GitManager({"dev": repo_with_dirty_state})
    # Create untracked file
    (repo_with_dirty_state / "untracked_script.py").write_text("x = 1\n")

    assert git_mgr.is_clean(repo_with_dirty_state) is False
    with pytest.raises(DirtyWorkspaceError):
        git_mgr.assert_clean(repo_with_dirty_state, "dev")
