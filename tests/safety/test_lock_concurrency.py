"""Safety: Concurrency Locking Tests (test_lock_concurrency.py).

Verifies:
1. Mutual exclusion: prevents concurrent backtests or agent runs on the same branch or GPU.
2. Safe automatic release via context manager on both success and exceptions.
3. Pruning of stale locks from terminated processes.
"""
import os
from pathlib import Path
import pytest

from src.core.lock_manager import LockManager, ConcurrencyLockError


def test_lock_acquire_and_release(tmp_path):
    lm = LockManager(lock_dir=tmp_path / "locks")
    assert lm.is_locked("claude") is False

    lm.acquire("claude")
    assert lm.is_locked("claude") is True

    # Acquiring again while locked must raise ConcurrencyLockError
    with pytest.raises(ConcurrencyLockError) as exc_info:
        lm.acquire("claude")
    assert "currently locked" in str(exc_info.value)

    lm.release("claude")
    assert lm.is_locked("claude") is False


def test_lock_context_manager_cleanup_on_exception(tmp_path):
    lm = LockManager(lock_dir=tmp_path / "locks")

    with pytest.raises(ValueError):
        with lm.lock("eval_runner"):
            assert lm.is_locked("eval_runner") is True
            raise ValueError("Simulated unexpected failure")

    # Lock must be released cleanly despite exception
    assert lm.is_locked("eval_runner") is False


def test_stale_lock_pruning(tmp_path):
    lm = LockManager(lock_dir=tmp_path / "locks")
    lock_file = tmp_path / "locks" / "dead_proc.lock"
    # Write non-existent PID (e.g. 99999999)
    lock_file.write_text("99999999")

    # is_locked should detect process is dead, prune lock file, and return False
    assert lm.is_locked("dead_proc") is False
    assert not lock_file.exists()
