"""Git branch and multi-workspace synchronization manager."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class GitOperationError(Exception):
    """General git command failure."""
    pass


class DirtyWorkspaceError(Exception):
    """Raised when a workspace has uncommitted changes that block execution or rebase."""
    pass


class RebaseConflictError(Exception):
    """Raised when git rebase hits a merge conflict and is automatically rolled back."""
    pass


class GitManager:
    """Manages Git branch states across model workspaces and handles golden main synchronizations."""

    def __init__(self, workspaces: Dict[str, Path], golden_branch: str = "main"):
        """
        workspaces: dict mapping workspace key ('gemini', 'claude', 'chatgpt') to its directory Path.
        golden_branch: name of the golden truth branch (default 'main').
        """
        self.workspaces = {k: Path(v).resolve() for k, v in workspaces.items()}
        self.golden_branch = golden_branch

    def _run_git(self, ws_path: Path, args: List[str], check: bool = True) -> str:
        cmd = ["git", "-C", str(ws_path)] + args
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=check,
                text=True,
            )
            return res.stdout.strip()
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() or e.stdout.strip()
            raise GitOperationError(f"Git command failed ({' '.join(cmd)}): {err_msg}") from e

    def get_current_branch(self, ws_path: Path) -> str:
        """Returns the currently checked-out branch name."""
        return self._run_git(ws_path, ["rev-parse", "--abbrev-ref", "HEAD"])

    def get_head_sha(self, ws_path: Path) -> str:
        """Returns the full SHA of HEAD."""
        return self._run_git(ws_path, ["rev-parse", "HEAD"])

    def get_commit_message(self, ws_path: Path, ref: str = "HEAD") -> str:
        """Returns the commit subject of the given ref."""
        return self._run_git(ws_path, ["log", "-1", "--pretty=%s", ref])

    def is_clean(self, ws_path: Path) -> bool:
        """Checks if the working tree has no uncommitted or untracked changes."""
        status = self._run_git(ws_path, ["status", "--porcelain"])
        return len(status.strip()) == 0

    def assert_clean(self, ws_path: Path, ws_name: str = "") -> None:
        """Raises DirtyWorkspaceError if the working tree is dirty."""
        if not self.is_clean(ws_path):
            status = self._run_git(ws_path, ["status", "--porcelain"])
            name_str = f" '{ws_name}'" if ws_name else ""
            raise DirtyWorkspaceError(
                f"Workspace{name_str} at {ws_path} has uncommitted changes:\n{status}\n"
                f"Cannot proceed until the workspace is clean."
            )

    def check_all_clean(self) -> List[str]:
        """Returns list of workspace keys that are dirty."""
        dirty = []
        for key, path in self.workspaces.items():
            if not self.is_clean(path):
                dirty.append(key)
        return dirty

    def has_new_commit(self, ws_path: Path, baseline_sha: str) -> bool:
        """Checks if HEAD is different from baseline_sha."""
        current_sha = self.get_head_sha(ws_path)
        return current_sha != baseline_sha

    def promote_to_main(
        self,
        candidate_ws: Path,
        golden_ws: Path,
        proposal_id: str,
        title: str,
        commit_sha: Optional[str] = None,
    ) -> str:
        """
        Promotes winning candidate commit into the golden main branch.
        Can merge from candidate branch/commit into main.
        Returns the new main commit SHA.
        """
        self.assert_clean(golden_ws, "golden_truth")
        if commit_sha is None:
            commit_sha = self.get_head_sha(candidate_ws)

        # Checkout main on golden_ws
        current_golden_branch = self.get_current_branch(golden_ws)
        if current_golden_branch != self.golden_branch:
            self._run_git(golden_ws, ["checkout", self.golden_branch])

        # Fetch candidate commit into golden repo if separate repos, or merge directly
        if golden_ws.resolve() != candidate_ws.resolve():
            self._run_git(golden_ws, ["fetch", str(candidate_ws), commit_sha])
            merge_target = "FETCH_HEAD"
        else:
            merge_target = commit_sha

        merge_msg = f"feat(harness): {proposal_id} - {title}\n\nPromoted from {commit_sha}"
        self._run_git(golden_ws, ["merge", "--no-ff", "-m", merge_msg, merge_target])
        new_main_sha = self.get_head_sha(golden_ws)
        return new_main_sha

    def rebase_branch_with_rollback(self, ws_path: Path, upstream_ref: str) -> bool:
        """
        Attempts to rebase the current branch onto upstream_ref.
        If a conflict occurs, executes `git rebase --abort` and raises RebaseConflictError.
        Returns True on success.
        """
        self.assert_clean(ws_path)
        cmd = ["git", "-C", str(ws_path), "rebase", upstream_ref]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            # Check if in rebase conflict state
            subprocess.run(["git", "-C", str(ws_path), "rebase", "--abort"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            err_msg = res.stderr.strip() or res.stdout.strip()
            raise RebaseConflictError(
                f"Rebase conflict detected while rebasing {ws_path} onto {upstream_ref}!\n"
                f"Rebase was aborted cleanly. Detail: {err_msg}"
            )
        return True

    def sync_all_workspaces(self, golden_ws: Path, skip_key: Optional[str] = None) -> Dict[str, bool]:
        """
        Synchronizes all model workspaces by rebasing or merging them from golden main.
        Returns a dict of {ws_key: True/False success}.
        """
        golden_main_sha = self.get_head_sha(golden_ws)
        results = {}

        for key, path in self.workspaces.items():
            if key == skip_key:
                results[key] = True
                continue

            if not path.exists():
                results[key] = False
                continue

            self.assert_clean(path, key)
            if path.resolve() != golden_ws.resolve():
                self._run_git(path, ["fetch", str(golden_ws), self.golden_branch])
                upstream = "FETCH_HEAD"
            else:
                upstream = self.golden_branch

            # Rebase onto the new golden truth
            self.rebase_branch_with_rollback(path, upstream)
            results[key] = True

        return results
