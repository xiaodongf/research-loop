"""Tier 5 Live Smoke Suite (@pytest.mark.live_smoke):
Verifies read-only 5-checkout telemetry, real FTS5 knowledge sync, real 10-metric CSV parsing,
real trading-exp /tmp worktree audit round-trip, and live Streamlit /_stcore/health boot.
"""
from __future__ import annotations
import json
import math
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Dict, Tuple
import pytest
import yaml

from src.bus.mcp_server import BusService
from src.bus.schema import AgentId, Intent, ThreadStatus
from src.harness.csv_parser import CSVParser, InvalidScorecardCSVError

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _snapshot_real_repos(workspaces: Dict[str, dict]) -> Dict[str, Tuple[str, str]]:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    snap: Dict[str, Tuple[str, str]] = {}
    for aid, conf in workspaces.items():
        p = Path(conf["path"])
        if not p.exists():
            continue
        head = subprocess.run(
            ["git", "-C", str(p), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(p), "status", "--porcelain"],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            check=True,
        ).stdout
        snap[aid] = (head, status)
    return snap


def _build_isolated_live_bus(tmp_path: Path) -> tuple[BusService, dict]:
    raw_cfg = yaml.safe_load((ROOT_DIR / "config.yaml").read_text(encoding="utf-8"))
    svc = BusService(
        root_dir=tmp_path,
        config=raw_cfg,
        db_path=tmp_path / "state" / "hub.db",
        peer_refs_dir=tmp_path / "state" / "peer_refs.git",
        worktree_root=Path("/tmp") / f"rl-live-smoke-{os.getpid()}",
    )
    return svc, raw_cfg


@pytest.mark.live_smoke
def test_gate_l1_live_five_checkout_telemetry_knowledge_and_csvs(tmp_path: Path) -> None:
    svc, raw_cfg = _build_isolated_live_bus(tmp_path)
    ws_map = raw_cfg["workspaces"]
    before = _snapshot_real_repos(ws_map)
    assert len(before) == 5

    # 1. Fleet telemetry across all 5 real checkouts
    fleet = {f.agent: f for f in svc.git_bridge.get_fleet_telemetry(store=svc.store)}
    assert set(fleet.keys()) == {
        AgentId.MAIN,
        AgentId.STAGING,
        AgentId.DEV,
        AgentId.CLAUDE,
        AgentId.CHATGPT,
    }
    for st_item in fleet.values():
        assert st_item.exists is True
        assert len(st_item.head_sha) == 40
    # Verify trading-gpt drift is accurately detected
    assert fleet[AgentId.CHATGPT].behind_main > 0

    # 2. Real check_host_capacity.py integration
    cap = svc.capacity.get_capacity_telemetry()
    assert cap["supply"]["ncpu"] >= 1
    assert cap["supply"]["avail_mem_gib"] > 0

    # 3. Sync real research_notes/ & reports/ across all 5 checkouts into FTS5
    sync_stats = svc.knowledge.sync_all_workspaces(max_files_per_workspace=60)
    assert sync_stats["scanned"] > 0
    assert sync_stats["added"] > 0

    ranker_hits = svc.store.search_knowledge("Stage 1 path ranker")
    assert len(ranker_hits) >= 1
    assert any(h.source_agent == AgentId.CLAUDE for h in ranker_hits)

    # 4. Real 10-metric A/B CSV parsing + real non-A/B trade dump rejection
    res_dir = Path(raw_cfg["evaluation"]["results_dir"])
    ab_csvs = sorted(res_dir.glob("*_ab-*_DEV_*.csv"))
    assert ab_csvs, f"Expected at least one *_ab-*_DEV_*.csv in {res_dir}"
    sc = CSVParser.parse_result_file(ab_csvs[-1])
    assert sc.num_folds >= 1
    for metric_val in (
        sc.base_pnl_total,
        sc.var_pnl_total,
        sc.delta_pnl_total,
        sc.base_win_rate,
        sc.var_win_rate,
        sc.fold_win_rate,
        sc.worst_fold_base,
        sc.worst_dd_base,
        sc.mean_dd_base,
        sc.mean_sharpe_base,
        sc.mean_exp_base,
        sc.base_prec1_mean,
    ):
        assert math.isfinite(metric_val)

    trade_dumps = sorted(res_dir.glob("all_cut_trades_*.csv"))
    if trade_dumps:
        with pytest.raises(InvalidScorecardCSVError):
            CSVParser.parse_result_file(trade_dumps[0])

    # 5. Zero-pollution proof: all 5 real trading* checkouts remain 100% unchanged
    after = _snapshot_real_repos(ws_map)
    assert after == before


@pytest.mark.live_smoke
def test_gate_l2_live_commit_worktree_and_readonly_audit_roundtrip(tmp_path: Path) -> None:
    svc, raw_cfg = _build_isolated_live_bus(tmp_path)
    ws_map = raw_cfg["workspaces"]
    before = _snapshot_real_repos(ws_map)

    main_ws = Path(ws_map["main"]["path"])
    head_sha = svc.git_bridge.resolve_commit(main_ws, "HEAD")
    base_sha = svc.git_bridge.resolve_commit(main_ws, "HEAD~1")

    # 1. Inspect real commit via peer_refs.git alternates (zero writes to trading-exp)
    diff_info = svc.git_bridge.inspect_commit(
        from_agent="main",
        thread_id="THR-LIVE",
        head_sha=head_sha,
        base_sha=base_sha,
    )
    assert diff_info["head_sha"] == head_sha
    assert len(diff_info["changed_files"]) >= 1

    # 2. Mount real disposable /tmp worktree and verify auto-cleanup
    with svc.git_bridge.disposable_worktree(head_sha=head_sha, thread_id="THR-LIVE", tag="smoke") as wt_dir:
        assert wt_dir.exists()
        assert (wt_dir / "AGENTS.md").exists()
    assert not wt_dir.exists()

    # 3. Full AUDIT_REQUEST -> AutoAuditWorker round-trip on real commit worktree
    live_llm = os.environ.get("RL_ENABLE_LIVE_LLM", "").strip() == "1"
    reviewer_target = os.environ.get("RL_LIVE_REVIEWER", "chatgpt")
    if not live_llm:
        # Sandboxed deterministic inspector executing inside the real /tmp worktree
        verdict_json = json.dumps(
            {
                "verdict": "APPROVED_WITH_CAUTION",
                "summary": f"Verified real commit {head_sha[:8]} in isolated /tmp worktree with AGENTS.md present.",
                "findings": [],
                "verified_finding_ids": [],
                "reopened_finding_ids": [],
            }
        )
        svc.audit_worker.provider_commands[reviewer_target] = (
            f"mock:test -f AGENTS.md && printf '%s\\n' '{verdict_json}'"
        )

    # Note: main_ws might have active local edits by the user today, so verify commit metadata without blocking on user's WIP
    code_ctx = svc.git_bridge.verify_commit_handoff(
        agent="main",
        head_sha=head_sha,
        base_sha=base_sha,
        workspace_path=main_ws,
        require_clean_changed_files=False,
    )
    req_env = svc.store.send_message(
        from_agent="main",
        to_agents=[reviewer_target],
        intent=Intent.AUDIT_REQUEST,
        subject=f"Smoke audit of real commit {head_sha[:8]}",
        body="Verify causality and contract invariants in disposable worktree.",
        code_context=code_ctx,
        auto_audit=True,
    )
    records = svc.audit_worker.execute_audit_for_message(req_env)
    assert len(records) == 1
    assert records[0].success is True, f"Audit failed: {records[0].error} | raw={records[0].raw_output[:300]}"
    bundle = svc.store.get_thread(req_env.thread_id)
    assert bundle is not None
    assert bundle["thread"].status in (ThreadStatus.APPROVED, ThreadStatus.CHANGES_REQUESTED)

    after = _snapshot_real_repos(ws_map)
    assert after == before


@pytest.mark.live_smoke
def test_gate_l3_live_streamlit_server_health_endpoint(tmp_path: Path) -> None:
    # Pick a guaranteed free localhost TCP port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    env = os.environ.copy()
    env["RL_DB_PATH"] = str(tmp_path / "ui_smoke_hub.db")
    env["HOME"] = str(tmp_path)

    proc = subprocess.Popen(
        ["bash", str(ROOT_DIR / "scripts" / "run_ui.sh"), str(port)],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        health_url = f"http://127.0.0.1:{port}/_stcore/health"
        deadline = time.monotonic() + 25.0
        healthy = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"Streamlit exited prematurely (code={proc.returncode}):\n{out}")
            try:
                with urllib.request.urlopen(health_url, timeout=1.5) as resp:
                    body = resp.read().decode("utf-8", errors="ignore").strip()
                    if resp.status == 200 and body == "ok":
                        healthy = True
                        break
            except Exception:
                time.sleep(0.3)
        assert healthy, f"Timed out waiting for {health_url} to return 200 'ok'"
    finally:
        if proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=5.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
