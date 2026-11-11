"""Tier 2 Tests: Zero-pollution Git provenance, dirty-reviewer safety, crash cleanup, and advisory-only guardrails."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict
import pytest

from src.bus.git_bridge import CommitNotFoundError, UncommittedHandoffError
from src.bus.mcp_server import BusService
from src.bus.schema import AgentId, Intent


def _snapshot_checkout_bytes(ws: Path) -> Dict[str, str]:
    """Captures SHA256 of every working-tree file plus git HEAD & status --porcelain."""
    snap: Dict[str, str] = {}
    head = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "-C", str(ws), "status", "--porcelain"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout
    snap["__HEAD__"] = head
    snap["__PORCELAIN__"] = porcelain
    for p in sorted(ws.rglob("*")):
        rel = p.relative_to(ws)
        if rel.parts and rel.parts[0] == ".git":
            continue
        if p.is_file():
            snap[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snap


def test_commit_provenance_verification_and_untracked_note_tolerance(five_repo_topology: Dict[str, Any]) -> None:
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])
    ws_dev: Path = five_repo_topology["checkouts"]["dev"]
    dev_sha: str = five_repo_topology["dev_feature_sha"]

    # 1. Non-existent commit SHA must raise CommitNotFoundError
    with pytest.raises(CommitNotFoundError):
        svc.git_bridge.verify_commit_handoff(agent="dev", head_sha="deadbeefdeadbeef")

    # 2. Unrelated untracked research note in sender workspace MUST be allowed
    untracked_note = ws_dev / "research_notes" / "wip_scratch.md"
    untracked_note.write_text("# Draft hypothesis notes\n", encoding="utf-8")
    ctx_ok = svc.git_bridge.verify_commit_handoff(
        agent="dev",
        head_sha=dev_sha,
        base_sha=five_repo_topology["main_head"],
    )
    assert ctx_ok.head_sha == dev_sha
    assert ctx_ok.changed_files == ["data_access/features.py"]
    assert untracked_note.exists()

    # 3. Uncommitted working-tree edits on a changed file in head_sha MUST raise UncommittedHandoffError
    feat_file = ws_dev / "data_access" / "features.py"
    original_txt = feat_file.read_text(encoding="utf-8")
    try:
        feat_file.write_text(original_txt + "# uncommitted drift!\n", encoding="utf-8")
        with pytest.raises(UncommittedHandoffError, match="uncommitted working-tree edits"):
            svc.git_bridge.verify_commit_handoff(
                agent="dev",
                head_sha=dev_sha,
                base_sha=five_repo_topology["main_head"],
            )
    finally:
        feat_file.write_text(original_txt, encoding="utf-8")


def test_dirty_reviewer_checkout_untouched_during_inspect_and_worktree_mount(
    five_repo_topology: Dict[str, Any],
) -> None:
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])
    ws_claude: Path = five_repo_topology["checkouts"]["claude"]
    dev_sha: str = five_repo_topology["dev_feature_sha"]
    main_sha: str = five_repo_topology["main_head"]

    # Put reviewer `claude` into a dirty state (uncommitted tracked edit + untracked research note)
    tracked_ranker = ws_claude / "models" / "ranker.py"
    tracked_ranker.write_text("def score(x):\n    return x * 2.5  # claude WIP edit\n", encoding="utf-8")
    untracked_study = ws_claude / "research_notes" / "claude_draft.md"
    untracked_study.write_text("Claude in-progress econometric derivation.\n", encoding="utf-8")

    before_claude = _snapshot_checkout_bytes(ws_claude)
    assert " M models/ranker.py" in before_claude["__PORCELAIN__"]
    assert "?? research_notes/claude_draft.md" in before_claude["__PORCELAIN__"]

    # Inspect dev's commit and mount a disposable worktree for claude's audit
    diff_info = svc.git_bridge.inspect_commit(
        from_agent="dev",
        thread_id="THR-0001",
        head_sha=dev_sha,
        base_sha=main_sha,
    )
    assert "parkinson_vol_lag1" in diff_info["diff_patch"]

    with svc.git_bridge.disposable_worktree(head_sha=dev_sha, thread_id="THR-0001", tag="claude") as wt_dir:
        assert wt_dir.exists()
        assert (wt_dir / "data_access" / "features.py").read_text(encoding="utf-8").find("parkinson_vol_lag1") != -1
        # Inside worktree, ranker.py is clean committed version, while ws_claude still has WIP edit!
        assert "claude WIP edit" not in (wt_dir / "models" / "ranker.py").read_text(encoding="utf-8")
        during_claude = _snapshot_checkout_bytes(ws_claude)
        assert during_claude == before_claude

    assert not wt_dir.exists()
    after_claude = _snapshot_checkout_bytes(ws_claude)
    assert after_claude == before_claude


def test_worktree_auto_cleanup_on_unhandled_exception(five_repo_topology: Dict[str, Any]) -> None:
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])
    dev_sha: str = five_repo_topology["dev_feature_sha"]

    captured_wt: Path | None = None
    with pytest.raises(RuntimeError, match="simulated crash"):
        with svc.git_bridge.disposable_worktree(head_sha=dev_sha, thread_id="THR-9999", tag="crash") as wt:
            captured_wt = wt
            assert wt.exists()
            raise RuntimeError("simulated crash")

    assert captured_wt is not None
    assert not captured_wt.exists()

    wt_list = subprocess.run(
        ["git", "-C", str(svc.git_bridge.peer_refs_dir), "worktree", "list", "--porcelain"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout
    assert str(captured_wt) not in wt_list


def test_advisory_only_drift_and_file_overlap_never_block(five_repo_topology: Dict[str, Any]) -> None:
    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])

    fleet = {f.agent: f for f in svc.git_bridge.get_fleet_telemetry(store=svc.store)}
    assert fleet[AgentId.CHATGPT].behind_main == 3
    assert fleet[AgentId.DEV].behind_main == 0
    assert fleet[AgentId.DEV].ahead_main == 1

    # Claude claims `data_access/features.py`
    clm_claude, adv_claude = svc.store.claim_work(
        agent="claude",
        files=["data_access/features.py"],
        summary="Testing intraday vol scaling",
    )
    assert clm_claude.active is True
    assert adv_claude == []

    # ChatGPT (3 behind main) claims the SAME file -> succeeds without blocking, returning BOTH overlap + drift advisories
    clm_gpt, adv_gpt = svc.store.claim_work(
        agent="chatgpt",
        files=["data_access/features.py"],
        summary="Inspecting feature bounds",
    )
    _, _, drift_adv = svc.git_bridge.get_agent_drift_advisory("chatgpt", store=svc.store)
    assert clm_gpt.active is True
    assert any("claude" in a and "data_access/features.py" in a for a in adv_gpt)
    assert drift_adv is not None and "3 commit(s) behind main" in drift_adv

    # Dev sends AUDIT_REQUEST touching `data_access/features.py` -> succeeds AND attaches overlap advisory on envelope
    send_res = svc.send_verified(
        from_agent="dev",
        to_agents=["claude"],
        intent=Intent.AUDIT_REQUEST,
        subject="Audit Parkinson vol feature",
        head_sha=five_repo_topology["dev_feature_sha"],
        base_sha=five_repo_topology["main_head"],
        auto_audit=False,
    )
    advisories = send_res["envelope"]["advisories"]
    assert any("active work claim overlapping" in a for a in advisories)
