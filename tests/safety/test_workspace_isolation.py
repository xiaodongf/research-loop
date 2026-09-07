import pytest, subprocess
from pathlib import Path
from src.core.isolation_guard import WorkspaceIsolationGuard, CrossWorkspaceContaminationError


def init_mock_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(path), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(path), check=True)
    (path / "README.md").write_text("# Initial repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=str(path), check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(path), check=True, stdout=subprocess.DEVNULL)
    return path


def test_workspace_isolation_normal_execution(tmp_path):
    ws_gemini = init_mock_git_repo(tmp_path / "trading")
    ws_claude = init_mock_git_repo(tmp_path / "trading-claude")
    ws_gpt = init_mock_git_repo(tmp_path / "trading-gpt")

    guard = WorkspaceIsolationGuard({
        "gemini": ws_gemini,
        "claude": ws_claude,
        "chatgpt": ws_gpt,
    })

    # Snapshot before agent runs
    snapshot = guard.take_snapshot()

    # Agent runs ONLY inside ws_claude
    (ws_claude / "new_feature.py").write_text("# new code\n")
    subprocess.run(["git", "add", "new_feature.py"], cwd=str(ws_claude), check=True)
    subprocess.run(["git", "commit", "-m", "feat: new feature"], cwd=str(ws_claude), check=True, stdout=subprocess.DEVNULL)

    # Verification must pass because ONLY claude changed
    guard.verify_isolation("claude", snapshot)


def test_cross_workspace_contamination_detected_and_rolled_back(tmp_path):
    ws_gemini = init_mock_git_repo(tmp_path / "trading")
    ws_claude = init_mock_git_repo(tmp_path / "trading-claude")
    ws_gpt = init_mock_git_repo(tmp_path / "trading-gpt")

    guard = WorkspaceIsolationGuard({
        "gemini": ws_gemini,
        "claude": ws_claude,
        "chatgpt": ws_gpt,
    })

    # Snapshot before agent runs
    snapshot = guard.take_snapshot()

    # Rogue action: Agent in claude accidentally writes a rogue file into ws_gemini
    (ws_gemini / "rogue_leak.py").write_text("# rogue write\n")

    # Verification MUST fail with CrossWorkspaceContaminationError
    with pytest.raises(CrossWorkspaceContaminationError) as exc_info:
        guard.verify_isolation("claude", snapshot)

    assert "Workspace 'gemini'" in str(exc_info.value)

    # Asserts that the rogue write in ws_gemini was cleaned up
    assert not (ws_gemini / "rogue_leak.py").exists()
