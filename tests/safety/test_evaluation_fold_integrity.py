import pytest, subprocess
from pathlib import Path
from src.core.isolation_guard import WorkspaceIsolationGuard, EvaluationFoldTamperingError


def init_mock_trading_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(path), check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(path), check=True)
    
    (path / "data_access").mkdir()
    (path / "data_access" / "features.py").write_text("# features\n")
    (path / "evaluation").mkdir()
    (path / "evaluation" / "folds.py").write_text("# protected folds\n")
    
    subprocess.run(["git", "add", "."], cwd=str(path), check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(path), check=True, stdout=subprocess.DEVNULL)
    return path


def test_valid_feature_modification_passes_integrity(tmp_path):
    ws = init_mock_trading_repo(tmp_path / "trading-claude")
    guard = WorkspaceIsolationGuard({"claude": ws})

    # Legitimate change to feature file
    (ws / "data_access" / "features.py").write_text("# features updated\n")
    subprocess.run(["git", "add", "."], cwd=str(ws), check=True)
    subprocess.run(["git", "commit", "-m", "feat: update features"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)

    # Must pass integrity check
    guard.verify_evaluation_integrity(ws)


def test_evaluation_fold_tampering_is_blocked(tmp_path):
    ws = init_mock_trading_repo(tmp_path / "trading-claude")
    guard = WorkspaceIsolationGuard({"claude": ws})

    # Rogue change: agent modifies evaluation/folds.py
    (ws / "evaluation" / "folds.py").write_text("# hacked folds\n")
    subprocess.run(["git", "add", "."], cwd=str(ws), check=True)
    subprocess.run(["git", "commit", "-m", "fix: alter fold dates"], cwd=str(ws), check=True, stdout=subprocess.DEVNULL)

    # Must raise EvaluationFoldTamperingError
    with pytest.raises(EvaluationFoldTamperingError) as exc_info:
        guard.verify_evaluation_integrity(ws)

    assert "evaluation/folds.py" in str(exc_info.value)
