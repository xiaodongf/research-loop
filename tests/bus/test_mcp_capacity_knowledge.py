"""Tier 3 Tests: Subprocess JSON-RPC 2.0 MCP server, 2-process atomic capacity lease race, and 5-checkout FTS5 knowledge sync."""
from __future__ import annotations
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from scripts.install_agent_configs import generate_bundle
from src.bus.capacity_lease import CapacityLeaseManager
from src.bus.mcp_server import BusService
from src.bus.store import BusStore

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _capacity_race_worker(
    db_path_str: str,
    agent_name: str,
    barrier: Any,
    out_q: Any,
) -> None:
    try:
        store = BusStore(db_path=Path(db_path_str))
        # Mock host supply: 32 cores, load=2.0 -> 30 idle cores; 52 GiB RAM.
        # One 4-worker job needs 24 + 4 reserve = 28 cores and 32 + 16 reserve = 48 GiB -> FITS ONE ONLY!
        mgr = CapacityLeaseManager(
            store,
            capacity_script_path=None,
            host_probe=lambda: {
                "ncpu": 32,
                "load1": 2.0,
                "load5": 2.0,
                "avail_mem_gib": 52.0,
                "running_jobs": [],
            },
        )
        barrier.wait(timeout=10.0)
        res = mgr.acquire_eval_lease(
            agent=agent_name,
            strategy="standalone_tb",
            mode="quick",
            workers=4.0,
            threads_per_worker=6.0,
            mem_per_worker_gib=8.0,
        )
        out_q.put(
            {
                "agent": agent_name,
                "acquired": res.acquired,
                "lease_id": res.lease.lease_id if res.lease else None,
                "wait_pos": res.waitlist_position,
                "wait_id": res.wait_id,
                "error": None,
            }
        )
    except Exception as exc:
        out_q.put({"agent": agent_name, "acquired": False, "error": repr(exc)})


def test_real_subprocess_mcp_stdio_jsonrpc_and_identity_spoofing_guard(
    five_repo_topology: Dict[str, Any],
) -> None:
    cfg_path: Path = five_repo_topology["config_path"]
    ws_claude: Path = five_repo_topology["checkouts"]["claude"]

    proc = subprocess.Popen(
        [
            sys.executable,
            str(ROOT_DIR / "src" / "bus" / "mcp_server.py"),
            "--agent",
            "claude",
            "--workspace",
            str(ws_claude),
            "--config",
            str(cfg_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def rpc(req_id: int, method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        line = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        proc.stdin.write(line + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]
        resp_line = proc.stdout.readline()  # type: ignore[union-attr]
        assert resp_line, "MCP server closed stdout unexpectedly"
        return json.loads(resp_line)

    try:
        # 1. initialize handshake
        init_resp = rpc(1, "initialize", {"protocolVersion": "2024-11-05"})
        assert init_resp["result"]["serverInfo"]["name"] == "research-loop-bus-claude"

        # 2. tools/list returns all 8 bus_* tools
        tools_resp = rpc(2, "tools/list")
        tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
        assert {
            "bus_status",
            "bus_send",
            "bus_reply",
            "bus_read_thread",
            "bus_inspect_commit",
            "bus_claim_work",
            "bus_acquire_eval_lease",
            "bus_knowledge",
        }.issubset(tool_names)

        # 3. Identity spoofing attempt (claude trying to send as main) MUST fail with IdentitySpoofingError
        spoof_resp = rpc(
            3,
            "tools/call",
            {
                "name": "bus_send",
                "arguments": {
                    "from_agent": "main",
                    "to_agents": ["dev"],
                    "intent": "AUDIT_REQUEST",
                    "subject": "Spoofed message",
                },
            },
        )
        assert spoof_resp["result"]["isError"] is True
        err_payload = json.loads(spoof_resp["result"]["content"][0]["text"])
        assert err_payload["error"] == "IdentitySpoofingError"

        # 4. Valid bus_send from claude -> dev & chatgpt
        send_resp = rpc(
            4,
            "tools/call",
            {
                "name": "bus_send",
                "arguments": {
                    "to_agents": ["dev", "chatgpt"],
                    "intent": "PROPOSAL_RFC",
                    "subject": "RFC: Intraday regime gate",
                    "body": "Please critique regime boundary stability.",
                    "auto_audit": False,
                },
            },
        )
        assert send_resp["result"]["isError"] is False
        sent_data = json.loads(send_resp["result"]["content"][0]["text"])
        assert sent_data["envelope"]["from_agent"] == "claude"
        tid = sent_data["envelope"]["thread_id"]

        # 5. bus_read_thread & bus_status
        thr_resp = rpc(5, "tools/call", {"name": "bus_read_thread", "arguments": {"thread_id": tid}})
        assert thr_resp["result"]["isError"] is False
        thr_data = json.loads(thr_resp["result"]["content"][0]["text"])
        assert thr_data["thread"]["thread_id"] == tid

        st_resp = rpc(6, "tools/call", {"name": "bus_status", "arguments": {}})
        st_data = json.loads(st_resp["result"]["content"][0]["text"])
        assert st_data["agent"] == "claude"
    finally:
        proc.stdin.close()
        proc.wait(timeout=5.0)


def test_two_process_barrier_capacity_lease_eliminates_toctou_and_reclaims_dead_pid(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "hub.db"
    store = BusStore(db_path=db_path)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    out_q = ctx.Queue()

    p_dev = ctx.Process(target=_capacity_race_worker, args=(str(db_path), "dev", barrier, out_q))
    p_claude = ctx.Process(target=_capacity_race_worker, args=(str(db_path), "claude", barrier, out_q))
    p_dev.start()
    p_claude.start()

    outcomes = [out_q.get(timeout=20.0), out_q.get(timeout=20.0)]
    p_dev.join(timeout=10.0)
    p_claude.join(timeout=10.0)

    for o in outcomes:
        assert o["error"] is None

    winners = [o for o in outcomes if o["acquired"]]
    waiters = [o for o in outcomes if not o["acquired"]]
    assert len(winners) == 1, f"Expected exactly 1 lease holder, got {outcomes}"
    assert len(waiters) == 1
    assert waiters[0]["wait_pos"] == 1

    # Release winner's lease and test dead-PID reclamation + ignore_host_guard + ab_partial progress
    mgr = CapacityLeaseManager(
        store,
        capacity_script_path=None,
        host_probe=lambda: {
            "ncpu": 32,
            "load1": 2.0,
            "load5": 2.0,
            "avail_mem_gib": 52.0,
            "running_jobs": [],
        },
    )
    assert mgr.release_eval_lease(winners[0]["lease_id"]) is True

    # Acquire a lease tied to a guaranteed dead PID (99_999_999), then acquire from staging -> dead lease auto-reclaimed!
    cp_dir = tmp_path / "ab_partial_test"
    cp_dir.mkdir()
    (cp_dir / "fold_01.json").write_text('{"fold": "2025-01", "status": "ok"}', encoding="utf-8")
    (cp_dir / "fold_02.json").write_text('{"fold": "2025-04", "error": "oom"}', encoding="utf-8")
    assert mgr.read_ab_partial_progress(str(cp_dir)) == {"completed_folds": 1, "failed_folds": 1}

    dead_lease = mgr.acquire_eval_lease(agent="chatgpt", pid=99_999_999, workers=4.0)
    assert dead_lease.acquired is True

    reclaimed_res = mgr.acquire_eval_lease(agent="staging", workers=4.0, checkpoint_dir=str(cp_dir))
    assert reclaimed_res.acquired is True

    # While staging's live PID holds the 4-worker lease, dev with ignore_host_guard=True can force override with advisory
    override_res = mgr.acquire_eval_lease(agent="dev", workers=4.0, ignore_host_guard=True)
    assert override_res.acquired is True
    assert any("ignore_host_guard=True" in a for a in override_res.advisories)


def test_hybrid_knowledge_sync_fts5_dedup_and_dry_run_installer(five_repo_topology: Dict[str, Any]) -> None:
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])
    ws_claude: Path = five_repo_topology["checkouts"]["claude"]
    ws_dev: Path = five_repo_topology["checkouts"]["dev"]

    # Create research_notes in claude and report in dev
    study_dir = ws_claude / "research_notes" / "Stage-1 path ranker study 2026-11"
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "synthesis.md").write_text(
        "# Stage-1 Path Ranker Synthesis\n\nCross-sectional monotonic barrier ranking improves tail precision.\n",
        encoding="utf-8",
    )
    (ws_dev / "reports" / "garman_klass_study.md").write_text(
        "# Garman-Klass Volatility Gate\n\nReduces high-vol bear drawdown by 2.4pp.\n",
        encoding="utf-8",
    )

    stats_1 = svc.knowledge.sync_all_workspaces()
    assert stats_1["added"] >= 2

    # Second sync must deduplicate 100% by SHA256
    stats_2 = svc.knowledge.sync_all_workspaces()
    assert stats_2["added"] == 0
    assert stats_2["deduped"] >= 2

    # Publish negative unmerged A/B experiment
    neg = svc.knowledge.publish_experiment_record(
        source_agent="chatgpt",
        title="Uncapped Focal Loss Gamma=4.0 on standalone_tb",
        summary="Degraded RECENT fold win rate from 60% to 35% due to extreme outlier overweighting.",
        outcome="NEGATIVE",
        strategy="standalone_tb",
        failure_conditions="gamma > 2.0 destabilizes calibration in bear regimes",
    )
    assert neg.category == "ab_negative"

    hits_ranker = svc.store.search_knowledge("Stage 1 path ranker")
    assert len(hits_ranker) >= 1
    assert hits_ranker[0].source_agent.value == "claude"

    hits_neg = svc.store.search_knowledge("Focal Loss Gamma", category="ab_negative")
    assert len(hits_neg) == 1
    assert hits_neg[0].outcome == "NEGATIVE"

    # Verify dry-run installer stages configs for all 5 checkouts WITHOUT writing into any checkout
    out_stage = five_repo_topology["hub_root"] / "staged_configs"
    actions = generate_bundle(
        root_dir=ROOT_DIR,
        config=five_repo_topology["config"],
        output_dir=out_stage,
        apply_to_workspaces=False,
    )
    assert len(actions) >= 10
    for aid in ("main", "staging", "dev", "claude", "chatgpt"):
        assert (out_stage / aid / ".mcp.json").exists()
        assert not (five_repo_topology["checkouts"][aid] / ".mcp.json").exists()
