"""Concurrency lock manager: prevents overlapping runs on the same workspace or evaluation engine."""
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Optional, Dict
from contextlib import contextmanager


class ConcurrencyLockError(Exception):
    """Raised when an operation is attempted on a locked workspace or runner."""
    pass


class LockManager:
    """Manages workspace and evaluation run locks."""

    def __init__(self, lock_dir: Optional[Path] = None):
        self.lock_dir = Path(lock_dir or (Path.cwd() / ".locks")).resolve()
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self._active_locks: Dict[str, Path] = {}

    def is_locked(self, resource_name: str) -> bool:
        lock_file = self.lock_dir / f"{resource_name}.lock"
        if not lock_file.exists():
            return False
        # Check if process holding lock is still alive
        try:
            pid_str = lock_file.read_text().strip()
            if pid_str:
                pid = int(pid_str)
                # os.kill(pid, 0) checks if process exists without killing it
                os.kill(pid, 0)
                return True
        except (ValueError, ProcessLookupError, PermissionError):
            # Stale lock
            lock_file.unlink(missing_ok=True)
            return False
        return False

    def acquire(self, resource_name: str) -> bool:
        if self.is_locked(resource_name):
            raise ConcurrencyLockError(
                f"Resource '{resource_name}' is currently locked by another running task. "
                f"Concurrent executions on the same branch or GPU are prohibited."
            )
        lock_file = self.lock_dir / f"{resource_name}.lock"
        lock_file.write_text(str(os.getpid()))
        self._active_locks[resource_name] = lock_file
        return True

    def release(self, resource_name: str) -> None:
        lock_file = self.lock_dir / f"{resource_name}.lock"
        if lock_file.exists():
            lock_file.unlink(missing_ok=True)
        self._active_locks.pop(resource_name, None)

    @contextmanager
    def lock(self, resource_name: str):
        self.acquire(resource_name)
        try:
            yield
        finally:
            self.release(resource_name)
