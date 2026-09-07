"""Abstract base class and models for in-process agent execution."""
from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any

from src.core.isolation_guard import WorkspaceIsolationGuard, EvaluationFoldTamperingError, CrossWorkspaceContaminationError
from src.core.git_manager import GitManager


@dataclass
class AgentExecutionResult:
    success: bool
    exit_code: int
    commit_sha: Optional[str] = None
    output: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0


class AgentRunner(ABC):
    """Abstract agent runner executing autonomously inside its workspace."""

    def __init__(
        self,
        name: str,
        workspace_path: Path,
        isolation_guard: Optional[WorkspaceIsolationGuard] = None,
        git_manager: Optional[GitManager] = None,
    ):
        self.name = name
        self.workspace_path = Path(workspace_path).resolve()
        self.isolation_guard = isolation_guard
        self.git_manager = git_manager or (
            GitManager({name: self.workspace_path}) if (self.workspace_path / ".git").exists() else None
        )

    def execute(
        self,
        prompt: str,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> AgentExecutionResult:
        """
        Executes the agent inside this workspace with pre/post safety verification and stdout streaming.
        """
        if not self.workspace_path.exists():
            raise FileNotFoundError(f"Workspace directory {self.workspace_path} does not exist.")

        # 1. Pre-execution snapshot
        snapshot = None
        initial_sha = None
        if self.isolation_guard:
            snapshot = self.isolation_guard.take_snapshot()

        if self.git_manager and (self.workspace_path / ".git").exists():
            try:
                initial_sha = self.git_manager.get_head_sha(self.workspace_path)
            except Exception:
                initial_sha = None

        start_time = time.time()

        # 2. Run agent process (subclass implementation)
        try:
            exit_code, full_output, error_msg = self._run_process(prompt, on_output)
        except Exception as e:
            duration = time.time() - start_time
            return AgentExecutionResult(
                success=False,
                exit_code=-1,
                output="",
                error=f"Agent process failed to launch or crashed: {e}",
                duration_seconds=duration,
            )

        duration = time.time() - start_time

        # 3. Post-execution isolation verification
        if self.isolation_guard and snapshot is not None:
            # Will raise CrossWorkspaceContaminationError and rollback rogue changes if any other workspace was touched
            self.isolation_guard.verify_isolation(active_key=self.name, snapshot=snapshot)

        # 4. Git commit detection and evaluation fold integrity verification
        commit_sha = None
        if self.git_manager and (self.workspace_path / ".git").exists():
            try:
                current_sha = self.git_manager.get_head_sha(self.workspace_path)
                if initial_sha and current_sha != initial_sha:
                    commit_sha = current_sha
                    # Check that the new commit did not touch immutable evaluation files
                    if self.isolation_guard:
                        self.isolation_guard.verify_evaluation_integrity(
                            self.workspace_path,
                            base_ref=initial_sha
                        )
            except EvaluationFoldTamperingError:
                raise
            except Exception as e:
                # Log error detecting commit
                pass

        success = (exit_code == 0)
        return AgentExecutionResult(
            success=success,
            exit_code=exit_code,
            commit_sha=commit_sha,
            output=full_output,
            error=error_msg,
            duration_seconds=duration,
        )

    @abstractmethod
    def _run_process(
        self,
        prompt: str,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> tuple[int, str, Optional[str]]:
        """
        Executes the specific agent command in self.workspace_path.
        Must return (exit_code, full_output, error_message).
        """
        pass
