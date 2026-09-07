"""ChatGPT workspace runner executing inside workspace/trading-gpt."""
from __future__ import annotations
import os, subprocess, select, time
from pathlib import Path
from typing import Callable, Optional, List

from src.agents.base import AgentRunner
from src.core.isolation_guard import WorkspaceIsolationGuard
from src.core.git_manager import GitManager


class GPTRunner(AgentRunner):
    """Executes ChatGPT / OpenAI Codex agent tasks inside workspace/trading-gpt."""

    DEFAULT_BINARY = "/usr/lib/chatgpt/resources/codex"

    def __init__(
        self,
        workspace_path: Path,
        binary_path: Optional[str] = None,
        command: Optional[str] = None,
        isolation_guard: Optional[WorkspaceIsolationGuard] = None,
        git_manager: Optional[GitManager] = None,
        timeout_seconds: int = 600,
    ):
        super().__init__(
            name="chatgpt",
            workspace_path=workspace_path,
            isolation_guard=isolation_guard,
            git_manager=git_manager,
        )
        self.binary_path = binary_path or self.DEFAULT_BINARY
        self.command = command
        self.timeout_seconds = timeout_seconds

    def _run_process(
        self,
        prompt: str,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> tuple[int, str, Optional[str]]:
        if self.command:
            cmd = ["bash", "-c", self.command]
        elif self.binary_path and self.binary_path.startswith("mock:"):
            cmd = ["bash", "-c", self.binary_path.replace("mock:", "")]
        elif self.binary_path and Path(self.binary_path).exists():
            cmd = [self.binary_path, "exec", "--dangerously-bypass-approvals-and-sandbox", prompt]
        else:
            cmd = ["python3", "-c", f"print('Executing ChatGPT task in {self.workspace_path}'); print({repr(prompt)})"]

        lines = []
        start_time = time.time()

        env = os.environ.copy()
        chatgpt_res = "/usr/lib/chatgpt/resources"
        if Path(chatgpt_res).exists() and chatgpt_res not in env.get("PATH", ""):
            env["PATH"] = f"{chatgpt_res}:{env.get('PATH', '')}"

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.workspace_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        try:
            while True:
                if time.time() - start_time > self.timeout_seconds:
                    proc.kill()
                    return -1, "\n".join(lines), f"ChatGPT execution timed out after {self.timeout_seconds}s"

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
