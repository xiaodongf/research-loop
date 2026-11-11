"""Tier 4A Tests: AutoAuditWorker read-only enforcement, 2-round remediation verification, and process-group killpg supervision."""
from __future__ import annotations
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict
import pytest

from src.bus.audit_worker import AutoAuditWorker, MutatingIntentForbiddenError
from src.bus.capacity_lease import is_pid_alive
from src.bus.mcp_server import BusService
from src.bus.schema import AgentId, AuditVerdictType, FindingStatus, Intent, ThreadStatus


def test_auto_audit_worker_rejects_mutating_intents(five_repo_topology: Dict[str, Any]) -> None:
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])

    for forbidden_intent in (
        Intent.INTEGRATION_REQUEST,
        Intent.ADVISORY_NOTE,
        Intent.ESCALATE_HUMAN,
    ):
        env = svc.store.send_message(
            from_agent="dev",
            to_agents=["staging"],
            intent=forbidden_intent,
            subject=f"Test {forbidden_intent.value}",
            auto_audit=False,
        )
        with pytest.raises(MutatingIntentForbiddenError, match="strictly restricted to read-only intents"):
            svc.audit_worker.execute_audit_for_message(env)


def test_two_round_headless_audit_remediation_and_verification_flow(
    five_repo_topology: Dict[str, Any],
) -> None:
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])
    ws_dev: Path = five_repo_topology["checkouts"]["dev"]
    sha1: str = five_repo_topology["dev_feature_sha"]
    main_sha: str = five_repo_topology["main_head"]

    # Round 1 mock providers:
    # - claude inspects the disposable worktree file and flags a P1 finding
    # - chatgpt returns APPROVED_WITH_CAUTION
    r1_claude_json = json.dumps(
        {
            "verdict": "CHANGES_REQUESTED",
            "summary": "Parkinson volatility estimator missing clip floor on zero-range bars.",
            "findings": [
                {
                    "severity": "P1",
                    "category": "numerical_stability",
                    "file": "data_access/features.py",
                    "lines": "1",
                    "problem": "Zero high-low range produces log(0) without epsilon floor.",
                    "suggested_fix": "Clip (high/low) ratio with 1e-8 floor.",
                }
            ],
            "verified_finding_ids": [],
            "reopened_finding_ids": [],
        }
    )
    r1_gpt_json = json.dumps(
        {
            "verdict": "APPROVED_WITH_CAUTION",
            "summary": "Lag-1 shift alignment looks clean; fix Claude's numerical floor note.",
            "findings": [],
            "verified_finding_ids": [],
            "reopened_finding_ids": [],
        }
    )

    svc.audit_worker.provider_commands = {
        "claude": f"mock:test -f data_access/features.py && printf '%s\\n' {shlex.quote(r1_claude_json)}",
        "chatgpt": f"mock:test -f data_access/features.py && printf '%s\\n' {shlex.quote(r1_gpt_json)}",
    }

    res_r1 = svc.send_verified(
        from_agent="dev",
        to_agents=["claude", "chatgpt"],
        intent=Intent.AUDIT_REQUEST,
        subject="Audit Parkinson feature commit",
        body="Please review numerical stability and lag alignment.",
        head_sha=sha1,
        base_sha=main_sha,
        run_auto_audit_now=True,
    )
    tid = res_r1["envelope"]["thread_id"]
    assert len(res_r1["auto_audit_results"]) == 2
    assert all(r["success"] for r in res_r1["auto_audit_results"])

    bundle_r1 = svc.store.get_thread(tid)
    assert bundle_r1 is not None
    assert bundle_r1["thread"].status == ThreadStatus.CHANGES_REQUESTED
    assert len(bundle_r1["findings"]) == 1
    fid = bundle_r1["findings"][0].id
    assert fid == f"{tid}-F1"
    assert bundle_r1["findings"][0].status == FindingStatus.OPEN

    # Dev commits fix sha2 in ws_dev and sends REMEDIATION_READY to claude
    (ws_dev / "data_access" / "features.py").write_text(
        "FEATURES = ['ret_1d', 'lag_1', 'lag_2', 'lag_3', 'parkinson_vol_lag1_clipped']\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(ws_dev), "add", "data_access/features.py"], check=True)
    subprocess.run(["git", "-C", str(ws_dev), "commit", "-m", "Clip Parkinson ratio floor 1e-8"], check=True)
    sha2 = subprocess.run(
        ["git", "-C", str(ws_dev), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()

    r2_claude_json = json.dumps(
        {
            "verdict": "APPROVED",
            "summary": "Verified 1e-8 clip floor in sha2 worktree.",
            "findings": [],
            "verified_finding_ids": [fid],
            "reopened_finding_ids": [],
        }
    )
    svc.audit_worker.provider_commands["claude"] = (
        f"mock:grep -q parkinson_vol_lag1_clipped data_access/features.py && printf '%s\\n' {shlex.quote(r2_claude_json)}"
    )

    res_r2 = svc.send_verified(
        from_agent="dev",
        thread_id=tid,
        to_agents=["claude"],
        intent=Intent.REMEDIATION_READY,
        subject="Fixed clip floor in sha2",
        head_sha=sha2,
        base_sha=sha1,
        resolved_findings={fid: sha2},
        run_auto_audit_now=True,
    )
    assert res_r2["auto_audit_results"][0]["verdict"] == "APPROVED"

    bundle_r2 = svc.store.get_thread(tid)
    assert bundle_r2 is not None
    assert bundle_r2["thread"].latest_round == 2
    assert bundle_r2["thread"].status == ThreadStatus.APPROVED
    assert bundle_r2["findings"][0].status == FindingStatus.VERIFIED
    assert bundle_r2["findings"][0].verified_by == AgentId.CLAUDE


def test_supervised_command_timeout_kills_entire_process_group(
    five_repo_topology: Dict[str, Any], tmp_path: Path
) -> None:
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])
    worker: AutoAuditWorker = svc.audit_worker
    pid_file = tmp_path / "grandchild.pid"

    # Spawn a bash shell that launches a detached background python sleeper & waits
    py_code = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    script = f"{shlex.quote(sys.executable)} -c {shlex.quote(py_code)} & wait"
    t0 = time.monotonic()
    code, _, err = worker.run_supervised_command(
        ["bash", "-c", script],
        cwd=tmp_path,
        timeout_seconds=0.5,
    )
    elapsed = time.monotonic() - t0
    assert code == -1
    assert err is not None and "timed out" in err
    assert elapsed < 4.0

    assert pid_file.exists(), "Grandchild process did not write its PID before timeout"
    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
    # Wait briefly for OS process table cleanup and verify grandchild was reaped by os.killpg
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and is_pid_alive(grandchild_pid):
        time.sleep(0.05)
    assert is_pid_alive(grandchild_pid) is False


def test_supervised_command_does_not_hang_when_stdin_reads_until_eof(
    five_repo_topology: Dict[str, Any], tmp_path: Path
) -> None:
    """
    Regression test for UI/non-TTY hang: `codex exec` reads stdin until EOF whenever stdin is a pipe.
    `run_supervised_command` must pass `stdin=subprocess.DEVNULL` so child processes see immediate EOF.
    """
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])
    worker: AutoAuditWorker = svc.audit_worker

    t0 = time.monotonic()
    code, out, err = worker.run_supervised_command(
        [sys.executable, "-c", "import sys; data = sys.stdin.read(); print(f'eof:{len(data)}')"],
        cwd=tmp_path,
        timeout_seconds=2.0,
    )
    elapsed = time.monotonic() - t0
    assert code == 0
    assert err is None
    assert "eof:0" in out
    assert elapsed < 1.5


def test_commitless_audit_request_auto_mounts_target_main_worktree_and_error_posts_note(
    five_repo_topology: Dict[str, Any],
) -> None:
    """
    Regression test: when `head_sha` is omitted on an AUDIT_REQUEST targeting `main`,
    AutoAuditWorker automatically mounts `main` HEAD in a disposable Git worktree instead
    of an empty temp directory, and any provider failure posts an [AUDIT_ERROR] note onto the thread.
    """
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])

    verdict_json = json.dumps(
        {
            "verdict": "APPROVED_WITH_CAUTION",
            "summary": "Audited mounted main HEAD worktree successfully.",
            "findings": [],
            "verified_finding_ids": [],
            "reopened_finding_ids": [],
        }
    )
    # Require `models/ranker.py` and `data_access/features.py` from `main` to exist inside cwd!
    svc.audit_worker.provider_commands = {
        "chatgpt": (
            f"mock:test -f models/ranker.py && test -f data_access/features.py "
            f"&& printf '%s\\n' {shlex.quote(verdict_json)}"
        ),
        "claude": "mock:echo 'simulated CLI auth failure' >&2 && exit 2",
    }

    env = svc.store.send_message(
        from_agent="chatgpt",
        to_agents=["chatgpt", "claude"],
        intent=Intent.AUDIT_REQUEST,
        subject="do an audit of the latest change on main branch on quantile strategy",
        body="Comprehensive audit of main branch.",
        code_context=None,
        auto_audit=True,
    )

    records = svc.audit_worker.execute_audit_for_message(env)
    assert len(records) == 2

    gpt_rec = next(r for r in records if r.reviewer == AgentId.CHATGPT)
    claude_rec = next(r for r in records if r.reviewer == AgentId.CLAUDE)
    assert gpt_rec.success is True
    assert gpt_rec.reply_envelope is not None
    assert gpt_rec.reply_envelope.verdict == AuditVerdictType.APPROVED_WITH_CAUTION
    assert claude_rec.success is False

    bundle = svc.store.get_thread(env.thread_id)
    assert bundle is not None
    subjects = [m.subject for m in bundle["messages"]]
    assert any("[APPROVED_WITH_CAUTION]" in s for s in subjects)
    assert any("[AUDIT_ERROR]" in s for s in subjects)
