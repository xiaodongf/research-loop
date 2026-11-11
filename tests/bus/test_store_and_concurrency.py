"""Tier 1 Tests: Explicit routing validation, 5-agent multicast inbox, finding lifecycle, and 5-process SQLite WAL stress."""
from __future__ import annotations
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from typing import List
import pytest
from pydantic import ValidationError

from src.bus.schema import (
    AgentId,
    AuditVerdictType,
    FindingInput,
    FindingSeverity,
    FindingStatus,
    Intent,
    MessageEnvelope,
    ThreadStatus,
)
from src.bus.store import BusStore
from src.harness.csv_parser import CSVParser, InvalidScorecardCSVError

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _concurrent_writer_worker(
    db_path_str: str,
    threads_dir_str: str,
    knowledge_dir_str: str,
    agent_name: str,
    barrier: Any,
    writes_per_proc: int,
    out_queue: Any,
) -> None:
    try:
        store = BusStore(
            db_path=Path(db_path_str),
            threads_dir=Path(threads_dir_str),
            knowledge_dir=Path(knowledge_dir_str),
        )
        barrier.wait(timeout=15.0)
        msg_ids: List[str] = []
        peers = [a for a in ("main", "staging", "dev", "claude", "chatgpt") if a != agent_name]
        for idx in range(writes_per_proc):
            target = peers[idx % len(peers)]
            env = store.send_message(
                from_agent=agent_name,
                to_agents=[target],
                intent=Intent.AUDIT_REQUEST,
                subject=f"Stress {agent_name} #{idx}",
                body=f"Payload {idx} from {agent_name}",
                auto_audit=False,
            )
            msg_ids.append(env.msg_id)
            if idx % 10 == 0:
                store.claim_work(
                    agent=agent_name,
                    files=[f"models/{agent_name}_{idx}.py", "data_access/shared.py"],
                    summary=f"Working batch {idx}",
                )
        out_queue.put({"agent": agent_name, "ok": True, "msg_ids": msg_ids, "error": None})
    except Exception as exc:
        out_queue.put({"agent": agent_name, "ok": False, "msg_ids": [], "error": repr(exc)})


def test_explicit_to_agents_required_across_schema_store_and_cli(tmp_path: Path) -> None:
    store = BusStore(db_path=tmp_path / "state" / "hub.db")

    # 1. Alias normalization ("gemini" -> AgentId.DEV)
    assert AgentId.normalize("gemini") == AgentId.DEV
    assert AgentId.normalize("DEV") == AgentId.DEV

    # 2. All 4 explicit-target intents must reject empty to_agents in both Pydantic & BusStore
    for req_intent in (
        Intent.AUDIT_REQUEST,
        Intent.PROPOSAL_RFC,
        Intent.REMEDIATION_READY,
        Intent.INTEGRATION_REQUEST,
    ):
        with pytest.raises((ValueError, ValidationError), match="explicit non-empty"):
            MessageEnvelope(
                msg_id="MSG-000001",
                thread_id="THR-0001",
                from_agent=AgentId.DEV,
                to_agents=[],
                intent=req_intent,
                subject="Missing target",
            )

        with pytest.raises(ValueError, match="explicit non-empty 'to_agents'"):
            store.send_message(
                from_agent="dev",
                to_agents=[],
                intent=req_intent,
                subject="Missing target",
            )

    # 3. CLI `bus send` without --to must exit with code 2 and write zero threads
    cli_res = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "cli.py"),
            "bus",
            "--db",
            str(store.db_path),
            "send",
            "--from",
            "dev",
            "--intent",
            "AUDIT_REQUEST",
            "--subject",
            "Forgot reviewer",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert cli_res.returncode == 2
    assert "explicit non-empty --to" in cli_res.stderr
    assert store.list_threads() == []


def test_five_agent_multicast_cursors_and_baseline_broadcast(tmp_path: Path) -> None:
    store = BusStore(db_path=tmp_path / "state" / "hub.db")

    env = store.send_message(
        from_agent="gemini",  # normalized to dev
        to_agents=["claude", "chatgpt"],
        cc_agents=["staging"],
        intent=Intent.AUDIT_REQUEST,
        subject="Audit ranker lag",
        body="Check shift(1)",
    )
    assert env.from_agent == AgentId.DEV

    counts = store.get_unread_counts()
    assert counts == {
        "main": 0,
        "staging": 1,
        "dev": 0,
        "claude": 1,
        "chatgpt": 1,
        "human": 0,
    }

    # Reading claude's inbox with mark_read=True must NOT clear chatgpt or staging
    claude_msgs = store.get_inbox("claude", mark_read=True)
    assert len(claude_msgs) == 1
    assert store.get_inbox("claude", unread_only=True) == []
    assert len(store.get_inbox("chatgpt", unread_only=True)) == 1
    assert len(store.get_inbox("staging", unread_only=True)) == 1

    # BASELINE_UPDATED from main with empty to_agents broadcasts to all 4 non-main workspaces
    store.send_message(
        from_agent="main",
        to_agents=[],
        intent=Intent.BASELINE_UPDATED,
        subject="main advanced to 51dec29",
    )
    counts_after = store.get_unread_counts()
    assert counts_after["main"] == 0
    assert counts_after["claude"] == 1
    assert counts_after["dev"] == 1
    assert counts_after["chatgpt"] == 2
    assert counts_after["staging"] == 2


def test_finding_lifecycle_and_atomic_markdown_mirror_parity(tmp_path: Path) -> None:
    store = BusStore(db_path=tmp_path / "state" / "hub.db")

    req = store.send_message(
        from_agent="dev",
        to_agents=["claude"],
        intent=Intent.AUDIT_REQUEST,
        subject="Volatility gate check",
        body="Initial commit",
    )
    tid = req.thread_id
    assert tid == "THR-0001"

    # Round 1: Claude files a blocking P0 finding
    v1 = store.send_message(
        from_agent="claude",
        to_agents=["dev"],
        thread_id=tid,
        in_reply_to=req.msg_id,
        intent=Intent.AUDIT_VERDICT,
        verdict=AuditVerdictType.CHANGES_REQUESTED,
        subject="Found lookahead leak",
        body="shift(1) missing on rolling std",
        new_findings=[
            FindingInput(
                severity=FindingSeverity.P0,
                category="lookahead",
                file="data_access/features.py",
                lines="42-48",
                problem="Unshifted rolling std uses T close at T open.",
                suggested_fix="Apply .shift(1) before barrier entry.",
            )
        ],
    )
    assert len(v1.findings) == 1
    fid = v1.findings[0].id
    assert fid == "THR-0001-F1"
    bundle_r1 = store.get_thread(tid)
    assert bundle_r1 is not None
    assert bundle_r1["thread"].status == ThreadStatus.CHANGES_REQUESTED
    assert bundle_r1["findings"][0].status == FindingStatus.OPEN

    # Round 2: Dev submits remediation commit resolving THR-0001-F1
    rem = store.send_message(
        from_agent="dev",
        to_agents=["claude"],
        thread_id=tid,
        in_reply_to=v1.msg_id,
        intent=Intent.REMEDIATION_READY,
        subject="Applied .shift(1)",
        resolved_findings={fid: "abc12345"},
    )
    assert rem.round_num == 2
    bundle_rem = store.get_thread(tid)
    assert bundle_rem is not None
    assert bundle_rem["thread"].status == ThreadStatus.AWAITING_AUDIT
    assert bundle_rem["findings"][0].status == FindingStatus.FIXED_PENDING_VERIFY
    assert bundle_rem["findings"][0].resolved_in_sha == "abc12345"

    # Claude verifies fix and approves
    store.send_message(
        from_agent="claude",
        to_agents=["dev"],
        thread_id=tid,
        in_reply_to=rem.msg_id,
        intent=Intent.AUDIT_VERDICT,
        verdict=AuditVerdictType.APPROVED,
        subject="Verified shift(1) fix",
        verified_findings=[fid],
    )
    bundle_final = store.get_thread(tid)
    assert bundle_final is not None
    assert bundle_final["thread"].status == ThreadStatus.APPROVED
    assert bundle_final["findings"][0].status == FindingStatus.VERIFIED
    assert bundle_final["findings"][0].verified_by == AgentId.CLAUDE

    # Check Markdown mirror parity and zero leftover .tmp files
    mirror_file = Path(bundle_final["mirror_path"])
    assert mirror_file.exists()
    mirror_txt = mirror_file.read_text(encoding="utf-8")
    assert "status: APPROVED" in mirror_txt
    assert "`THR-0001-F1`" in mirror_txt
    assert "`VERIFIED`" in mirror_txt
    assert list(store.threads_dir.glob(".*.tmp")) == []


def test_five_process_barrier_sqlite_wal_stress(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "hub.db"
    t_dir = tmp_path / "threads"
    k_dir = tmp_path / "knowledge"
    # Pre-initialize schema once, then hammer concurrently from 5 OS processes
    store = BusStore(db_path=db_path, threads_dir=t_dir, knowledge_dir=k_dir)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(5)
    out_q = ctx.Queue()
    writes_per_proc = 50
    agents = ["main", "staging", "dev", "claude", "chatgpt"]

    procs = [
        ctx.Process(
            target=_concurrent_writer_worker,
            args=(str(db_path), str(t_dir), str(k_dir), ag, barrier, writes_per_proc, out_q),
        )
        for ag in agents
    ]
    for p in procs:
        p.start()

    results = [out_q.get(timeout=45.0) for _ in agents]
    for p in procs:
        p.join(timeout=15.0)
        assert p.exitcode == 0

    for r in results:
        assert r["ok"], f"Worker {r['agent']} failed: {r['error']}"

    all_ids = [mid for r in results for mid in r["msg_ids"]]
    assert len(all_ids) == 5 * writes_per_proc  # 250 messages
    assert len(set(all_ids)) == 5 * writes_per_proc  # Zero duplicate IDs
    assert len(store.list_threads(limit=500)) == 250


def test_ten_metric_csv_parser_staged_and_strict_validation(tmp_path: Path) -> None:
    dev_csv = tmp_path / "standalone_tb_ab-parkinson_DEV_20261111_100000_7665697.csv"
    recent_csv = tmp_path / "standalone_tb_ab-parkinson_RECENT_20261111_100000_7665697.csv"

    header = (
        "fold,regime,spy,base_n,base_win,base_total_return,base_max_dd,base_sharpe,base_prec1,base_avg_exp,"
        "var_n,var_win,var_total_return,var_max_dd,var_sharpe,var_prec1,var_avg_exp\n"
    )
    dev_csv.write_text(
        header
        + "2024-01,BULL,4.2,20,0.50,8.0,-5.0,1.10,55.0,32.0,22,0.55,11.0,-4.0,1.35,60.0,34.0\n"
        + "2024-04,BEAR,-3.1,18,0.45,-2.0,-8.5,-0.20,48.0,28.0,18,0.50,1.5,-6.0,0.25,52.0,29.0\n",
        encoding="utf-8",
    )
    recent_csv.write_text(
        header
        + "2026-01,VOL,1.5,15,0.60,6.0,-4.5,1.20,58.0,30.0,16,0.625,9.5,-3.8,1.55,63.0,31.0\n",
        encoding="utf-8",
    )

    staged = CSVParser.parse_staged_run([dev_csv, recent_csv])
    assert set(staged.keys()) == {"DEV", "RECENT", "TOTAL"}
    tot = staged["TOTAL"]
    assert tot.strategy == "standalone_tb"
    assert tot.variant == "parkinson"
    assert tot.num_folds == 3
    assert tot.fold_wins == 3
    assert tot.fold_win_rate == 100.0
    assert tot.delta_pnl_total == pytest.approx((11.0 + 1.5 + 9.5) - (8.0 - 2.0 + 6.0))
    assert tot.worst_dd_base == pytest.approx(-8.5)
    assert tot.worst_dd_var == pytest.approx(-6.0)
    assert tot.total_trades_base == 53
    assert tot.total_trades_var == 56

    # Reject non-A/B trade dump CSVs and non-finite NaN values
    trade_dump = tmp_path / "all_cut_trades_20261111.csv"
    trade_dump.write_text("ticker,entry_date,exit_date,ret\nAAPL,2026-01-02,2026-01-10,0.04\n", encoding="utf-8")
    with pytest.raises(InvalidScorecardCSVError, match="missing mandatory columns"):
        CSVParser.parse_result_file(trade_dump)

    nan_csv = tmp_path / "standalone_tb_ab-bad_DEV_20261111_100000_7665697.csv"
    nan_csv.write_text(
        header + "2024-01,BULL,4.2,20,0.50,NaN,-5.0,1.10,55.0,32.0,22,0.55,11.0,-4.0,1.35,60.0,34.0\n",
        encoding="utf-8",
    )
    with pytest.raises(InvalidScorecardCSVError, match="Non-finite"):
        CSVParser.parse_result_file(nan_csv)
