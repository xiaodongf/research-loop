"""Claude Code CLI runner executing inside workspace/trading-claude."""
from __future__ import annotations
import os, subprocess, select, time
from pathlib import Path
from typing import Callable, Optional, List

from src.agents.base import AgentRunner
from src.core.isolation_guard import WorkspaceIsolationGuard
from src.core.git_manager import GitManager


class ClaudeCodeRunner(AgentRunner):
    """Executes Claude Code CLI headlessly inside workspace/trading-claude."""

    DEFAULT_BINARY = "/home/xiaodong/.config/Claude/claude-code/2.1.255/claude"

    def __init__(
        self,
        workspace_path: Path,
        binary_path: Optional[str] = None,
        isolation_guard: Optional[WorkspaceIsolationGuard] = None,
        git_manager: Optional[GitManager] = None,
        timeout_seconds: int = 600,
    ):
        super().__init__(
            name="claude",
            workspace_path=workspace_path,
            isolation_guard=isolation_guard,
            git_manager=git_manager,
        )
        self.binary_path = binary_path or self.DEFAULT_BINARY
        self.timeout_seconds = timeout_seconds

    def _run_process(
        self,
        prompt: str,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> tuple[int, str, Optional[str]]:
        # Check if binary exists or if custom command passed
        if not Path(self.binary_path).exists() and not self.binary_path.startswith("mock:"):
            # Check if claude is in PATH
            import shutil
            which_claude = shutil.which("claude")
            if which_claude:
                bin_cmd = which_claude
            else:
                bin_cmd = self.binary_path
        else:
            bin_cmd = self.binary_path

        if bin_cmd.startswith("mock:"):
            cmd = ["bash", "-c", bin_cmd.replace("mock:", "")]
        else:
            cmd = [bin_cmd, "-p", prompt, "--dangerously-skip-permissions"]

        lines = []
        start_time = time.time()

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.workspace_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        try:
            while True:
                if time.time() - start_time > self.timeout_seconds:
                    proc.kill()
                    return -1, "\n".join(lines), f"Claude Code execution timed out after {self.timeout_seconds}s"

                rlist, _, _ = select.select([proc.stdout], [], [], 0.1)
                if rlist:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    line_clean = line.rstrip("\n")
                    lines.append(line_clean)
                    if on_output:
                        on_output(line_clean)
                else:
                    if proc.poll() is not None:
                        # Flush remaining
                        for remaining in proc.stdout.readlines():
                            line_clean = remaining.rstrip("\n")
                            lines.append(line_clean)
                            if on_output:
                                on_output(line_clean)
                        break

            proc.wait()
            exit_code = proc.returncode
            full_output = "\n".join(lines)
            err = None if exit_code == 0 else f"Process exited with non-zero code {exit_code}"
            return exit_code, full_output, err

        except Exception as e:
            proc.kill()
            return -1, "\n".join(lines), str(e)
