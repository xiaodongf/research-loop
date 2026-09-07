"""Safety and isolation guard: prevents cross-workspace writes and protects evaluation folds."""
from __future__ import annotations
import os, subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any


class CrossWorkspaceContaminationError(Exception):
    """Raised when an agent running in one workspace modifies files in another workspace."""
    pass


class EvaluationFoldTamperingError(Exception):
    """Raised when an agent attempts to modify core evaluation files (e.g. evaluation/folds.py)."""
    pass


class WorkspaceIsolationGuard:
    IMMUTABLE_EVAL_FILES = [
        "evaluation/folds.py",
        "evaluation/runner.py",
        "evaluation/metrics.py",
    ]

    def __init__(self, workspaces: Dict[str, Path]):
        """
        workspaces: dict mapping workspace key ('gemini', 'claude', 'chatgpt') to its directory Path.
        """
        self.workspaces = {k: Path(v).resolve() for k, v in workspaces.items()}

    def get_workspace_state(self, ws_path: Path) -> Dict[str, Any]:
        if not ws_path.exists() or not (ws_path / ".git").exists():
            return {"head": "nogit", "dirty": "", "untracked": []}
        try:
            head = subprocess.check_output(
                ["git", "-C", str(ws_path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).strip().decode()
            status = subprocess.check_output(
                ["git", "-C", str(ws_path), "status", "--porcelain"],
                stderr=subprocess.DEVNULL
            ).strip().decode()
            return {"head": head, "status": status}
        except Exception as e:
            return {"head": "error", "error": str(e), "status": ""}

    def take_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Snapshots the Git state of all configured workspaces."""
        return {k: self.get_workspace_state(path) for k, path in self.workspaces.items()}

    def verify_isolation(self, active_key: str, snapshot: Dict[str, Dict[str, Any]]) -> None:
        """
        Verifies that ONLY active_key was modified.
        If any other workspace has changed, rolls back stray changes and raises CrossWorkspaceContaminationError.
        """
        breaches = []
        for key, path in self.workspaces.items():
            if key == active_key:
                continue

            current_state = self.get_workspace_state(path)
            prev_state = snapshot.get(key, {})

            if current_state["head"] != prev_state.get("head") or current_state["status"] != prev_state.get("status"):
                breaches.append(f"Workspace '{key}' ({path}) was modified unexpectedly! Status diff: {current_state['status']}")
                # Rollback stray changes in non-target workspace
                try:
                    subprocess.run(["git", "-C", str(path), "reset", "--hard", "HEAD"], check=False)
                    subprocess.run(["git", "-C", str(path), "clean", "-fd"], check=False)
                except Exception:
                    pass

        if breaches:
            msg = "\n".join(breaches)
            raise CrossWorkspaceContaminationError(
                f"Cross-workspace write contamination detected!\n{msg}\nAll rogue changes rolled back."
            )

    def verify_evaluation_integrity(self, target_ws_path: Path, base_ref: str = "HEAD~1") -> None:
        """
        Inspects the git diff of the active workspace to ensure immutable evaluation files were NOT modified.
        """
        try:
            diff_output = subprocess.check_output(
                ["git", "-C", str(target_ws_path), "diff", "--name-only", base_ref, "HEAD"],
                stderr=subprocess.DEVNULL
            ).strip().decode()
            changed_files = [f.strip() for f in diff_output.splitlines() if f.strip()]
        except Exception:
            # Fallback to checking uncommitted diff
            try:
                diff_output = subprocess.check_output(
                    ["git", "-C", str(target_ws_path), "status", "--porcelain"],
                    stderr=subprocess.DEVNULL
                ).strip().decode()
                changed_files = [line[3:].strip() for line in diff_output.splitlines() if line.strip()]
            except Exception:
                changed_files = []

        for f in changed_files:
            for immutable in self.IMMUTABLE_EVAL_FILES:
                if f == immutable or f.endswith(immutable):
                    raise EvaluationFoldTamperingError(
                        f"Evaluation integrity violation: Agent attempted to modify protected file '{f}'. "
                        f"Modifying evaluation folds, runners, or metrics is strictly forbidden."
                    )
