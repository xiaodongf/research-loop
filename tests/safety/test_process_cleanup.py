"""Safety: Process Cleanup & Cancellation Tests (test_process_cleanup.py).

Verifies:
1. Hard timeout terminates child subprocess cleanly (SIGTERM / SIGKILL).
2. Runner handles abnormal exits gracefully without hanging or leaving zombie processes.
"""
import time
from pathlib import Path
import pytest

from src.agents.claude_runner import ClaudeCodeRunner
from src.harness.eval_bridge import EvaluationBridge, EvaluationTimeoutError


def test_agent_runner_timeout_kill(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    # Spawns a sleep process with small timeout of 1 second
    runner = ClaudeCodeRunner(
        workspace_path=ws,
        binary_path="mock:sleep 10",
        timeout_seconds=1,
    )
    start = time.time()
    result = runner.execute("Run slow task")
    elapsed = time.time() - start

    assert result.success is False
    assert result.exit_code == -1
    assert "timed out" in (result.error or "")
    assert elapsed < 3.0  # Killed promptly without waiting 10s


def test_eval_bridge_timeout_kill(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    results = tmp_path / "results"
    results.mkdir()

    slow_script = tmp_path / "slow_eval.sh"
    slow_script.write_text("#!/bin/bash\nsleep 10\n")
    slow_script.chmod(0o755)

    bridge = EvaluationBridge(
        python_bin=slow_script,
        results_dir=results,
        default_timeout=1,
    )
    start = time.time()
    with pytest.raises(EvaluationTimeoutError) as exc_info:
        bridge.run_evaluation(workspace_path=ws, timeout=1)
    elapsed = time.time() - start

    assert "timed out" in str(exc_info.value)
    assert elapsed < 3.0
