"""Research Loop: 5-Agent Multi-Checkout Control Tower (Streamlit Web Dashboard)."""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import importlib
from src.agents.dispatcher import AgentDispatcher
import src.bus.audit_worker as _aw_mod
import src.bus.mcp_server as _mcp_mod

if not hasattr(_aw_mod, "launch_background_audit"):
    importlib.reload(_aw_mod)
    importlib.reload(_mcp_mod)

from src.bus.audit_worker import (
    launch_background_audit,
    list_audit_jobs,
    stop_audit_job,
)
from src.bus.mcp_server import BusService
from src.bus.schema import (
    EXPLICIT_TARGET_INTENTS,
    READ_ONLY_AUTO_INTENTS,
    AgentId,
    AuditVerdictType,
    FindingStatus,
    Intent,
    ThreadStatus,
)
from src.core.git_manager import GitManager
from src.core.ideation import IdeationEngine
from src.core.proposal_manager import ProposalManager
from src.core.schema import (
    ModelName,
    ProposalStatus,
    ScorecardSummary,
    StrategyTarget,
)
from src.core.state_machine import ProposalStateMachine
from src.harness.csv_parser import CSVParser, InvalidScorecardCSVError
from src.harness.eval_bridge import EvalBridge

_IS_APPTEST = ("pytest" in sys.modules) or ("streamlit.testing.v1.app_test" in sys.modules)


def resolve_runtime_paths() -> tuple[Path, Path, Optional[Path]]:
    cfg_env = os.environ.get("RL_CONFIG_PATH")
    db_env = os.environ.get("RL_DB_PATH")
    cfg_path = Path(cfg_env).resolve() if cfg_env else (ROOT_DIR / "config.yaml")
    eff_root = cfg_path.parent if cfg_env else ROOT_DIR
    db_path = Path(db_env).resolve() if db_env else None
    return eff_root, cfg_path, db_path


def load_config(cfg_path: Path) -> Dict[str, Any]:
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_ten_metric_dataframe(sc_data: Dict[str, Any] | ScorecardSummary) -> pd.DataFrame:
    d = sc_data.model_dump() if isinstance(sc_data, ScorecardSummary) else dict(sc_data)
    num_folds = int(d.get("num_folds", 0))
    fold_wins = int(d.get("fold_wins", 0))
    fold_wr = float(d.get("fold_win_rate", (fold_wins / num_folds * 100.0) if num_folds else 0.0))

    rows = [
        {
            "Metric": "1. Cumulative Return (pp)",
            "Baseline": f"{float(d.get('base_pnl_total', 0.0)):+.2f}%",
            "Variant": f"{float(d.get('var_pnl_total', 0.0)):+.2f}%",
            "Delta": f"{float(d.get('delta_pnl_total', 0.0)):+.2f}%",
        },
        {
            "Metric": "2. Return ex-Largest Fold (pp)",
            "Baseline": f"{float(d.get('base_pnl_ex_largest', 0.0)):+.2f}%",
            "Variant": f"{float(d.get('var_pnl_ex_largest', 0.0)):+.2f}%",
            "Delta": f"{float(d.get('delta_pnl_ex_largest', 0.0)):+.2f}%",
        },
        {
            "Metric": "3. Pooled Trade Win Rate",
            "Baseline": f"{float(d.get('base_win_rate', 0.0)):.2f}%",
            "Variant": f"{float(d.get('var_win_rate', 0.0)):.2f}%",
            "Delta": f"{float(d.get('delta_win_rate', 0.0)):+.2f}%",
        },
        {
            "Metric": "4. Paired Fold Win Rate",
            "Baseline": f"{num_folds} folds",
            "Variant": f"{fold_wins}/{num_folds} wins",
            "Delta": f"{fold_wr:.1f}%",
        },
        {
            "Metric": "5. Worst-Fold Return",
            "Baseline": f"{float(d.get('worst_fold_base', 0.0)):+.2f}%",
            "Variant": f"{float(d.get('worst_fold_var', 0.0)):+.2f}%",
            "Delta": f"{float(d.get('delta_worst_fold', 0.0)):+.2f}%",
        },
        {
            "Metric": "6. Worst-Fold Max Drawdown",
            "Baseline": f"{float(d.get('worst_dd_base', 0.0)):.2f}%",
            "Variant": f"{float(d.get('worst_dd_var', 0.0)):.2f}%",
            "Delta": f"{float(d.get('delta_worst_dd', 0.0)):+.2f}%",
        },
        {
            "Metric": "7. Mean Max Drawdown",
            "Baseline": f"{float(d.get('mean_dd_base', 0.0)):.2f}%",
            "Variant": f"{float(d.get('mean_dd_var', 0.0)):.2f}%",
            "Delta": f"{float(d.get('delta_mean_dd', 0.0)):+.2f}%",
        },
        {
            "Metric": "8. Mean Sharpe Ratio",
            "Baseline": f"{float(d.get('mean_sharpe_base', 0.0)):.3f}",
            "Variant": f"{float(d.get('mean_sharpe_var', 0.0)):.3f}",
            "Delta": f"{float(d.get('delta_mean_sharpe', 0.0)):+.3f}",
        },
        {
            "Metric": "9. Mean Exposure (%)",
            "Baseline": f"{float(d.get('mean_exp_base', 0.0)):.2f}%",
            "Variant": f"{float(d.get('mean_exp_var', 0.0)):.2f}%",
            "Delta": f"{float(d.get('delta_mean_exp', 0.0)):+.2f}%",
        },
        {
            "Metric": "10. Precision@10% & Trade Count",
            "Baseline": f"{float(d.get('base_prec1_mean', 0.0)):.2f}% (N={int(d.get('total_trades_base', 0))})",
            "Variant": f"{float(d.get('var_prec1_mean', 0.0)):.2f}% (N={int(d.get('total_trades_var', 0))})",
            "Delta": f"{float(d.get('delta_prec1_mean', 0.0)):+.2f}%",
        },
    ]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Page Initialization & Core Services
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Research Loop | 5-Agent Control Tower",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

runtime_root, cfg_path, override_db = resolve_runtime_paths()
config = load_config(cfg_path)
bus_svc = BusService(root_dir=runtime_root, config=config, db_path=override_db)
state_dir = bus_svc.store.db_path.parent
pm = ProposalManager(root_dir=runtime_root)

workspaces_conf = config.get("workspaces", {})
ws_paths = {
    k: Path(v["path"])
    for k, v in workspaces_conf.items()
    if isinstance(v, dict) and v.get("path") and Path(v["path"]).exists()
}
golden_key = config.get("golden_truth", {}).get("workspace", "main")
golden_entry = (
    workspaces_conf.get(golden_key)
    or workspaces_conf.get("main")
    or workspaces_conf.get("dev")
    or workspaces_conf.get("gemini")
    or {}
)
golden_path = Path(golden_entry["path"]) if isinstance(golden_entry, dict) and golden_entry.get("path") else None
git_mgr = (
    GitManager(workspaces=ws_paths, golden_branch=config.get("golden_truth", {}).get("branch", "main"))
    if ws_paths
    else None
)

eval_conf = config.get("evaluation", {})
results_dir = Path(eval_conf.get("results_dir", runtime_root / "evaluation" / "outputs" / "results"))
eval_bridge = EvalBridge(
    python_bin=Path(eval_conf.get("python_bin", sys.executable)),
    results_dir=results_dir,
)
dispatcher = AgentDispatcher(config=config, proposal_manager=pm, eval_bridge=eval_bridge)

fleet_statuses = bus_svc.git_bridge.get_fleet_telemetry(store=bus_svc.store)
fleet_by_name = {fs.agent.value: fs for fs in fleet_statuses}
cap_telemetry = bus_svc.capacity.get_capacity_telemetry()

# ----------------------------------------------------------------------
# Top Header & Live 5-Checkout Fleet + Host Capacity Banner
# ----------------------------------------------------------------------
st.title("🏛️ Research Loop: 5-Agent Trading Research Control Tower")
st.caption(
    "Zero-Pollution Coordination, Headless Read-Only Worktree Audits, Shared FTS5 Memory, and 10-Metric Walk-Forward Scorecards "
    "across `main`, `staging`, `dev`, `claude`, and `chatgpt`."
)

fleet_cols = st.columns(max(1, len(fleet_statuses)))
agent_icons = {
    "main": "👑",
    "staging": "🧪",
    "dev": "🔵",
    "claude": "🟣",
    "chatgpt": "🟢",
}

for idx, fs in enumerate(fleet_statuses):
    with fleet_cols[idx]:
        with st.container(border=True):
            icon = agent_icons.get(fs.agent.value, "⚪")
            clean_badge = "🟢 Clean" if fs.is_clean else f"🟡 Dirty ({fs.dirty_count})"
            st.markdown(f"**{icon} `{fs.agent.value}`** (`{fs.branch or fs.expected_branch}`)")
            if fs.exists:
                st.caption(f"`{fs.head_sha[:7]}` · {clean_badge} · 📬 `{fs.unread_count}` unread")
                if fs.behind_main > 0:
                    st.warning(f"Advisory: {fs.behind_main} behind main (+{fs.ahead_main} ahead)")
                else:
                    st.caption(f"Synced with main (+{fs.ahead_main} ahead)")
                if fs.active_claims:
                    claim_files = [f for c in fs.active_claims for f in c.files]
                    st.caption(f"📌 Active claims ({len(claim_files)}): `{', '.join(claim_files[:2])}`")
            else:
                st.error("Missing checkout path")

sup = cap_telemetry.get("supply", {})
active_leases = cap_telemetry.get("active_leases", [])
waitlist_items = cap_telemetry.get("waitlist", [])

hc1, hc2, hc3, hc4 = st.columns(4)
hc1.metric("Host CPU Load (1m / Cores)", f"{sup.get('load1', 0.0):.1f} / {sup.get('ncpu', 0)}")
hc2.metric("Available RAM", f"{sup.get('avail_mem_gib', 0.0):.1f} GiB")
hc3.metric("Active Eval Leases", f"{len(active_leases)}")
hc4.metric("Eval Waitlist", f"{len(waitlist_items)}")

# ----------------------------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------------------------
st.sidebar.header("🌿 Fleet & Sync Controls")
for fs in fleet_statuses:
    short_h = fs.head_sha[:7] if fs.head_sha else "N/A"
    drift_txt = f" | ⚠️ -{fs.behind_main} behind main" if fs.behind_main > 0 else ""
    st.sidebar.caption(f"**`{fs.agent.value}`** (`{fs.branch}`): `{short_h}`{drift_txt}")

if st.sidebar.button("🔄 Sync Knowledge Across 5 Checkouts", key="sidebar_sync_knowledge_btn", use_container_width=True):
    stats = bus_svc.knowledge.sync_all_workspaces()
    st.sidebar.success(
        f"Indexed {stats['scanned']} files (+{stats['added']} new, {stats['deduped']} deduped)"
    )

if st.sidebar.button("⚡ Rebase All Branches from Main", key="sidebar_rebase_btn", use_container_width=True):
    if git_mgr and golden_path and golden_path.exists():
        try:
            res = git_mgr.sync_all_workspaces(golden_ws=golden_path)
            st.sidebar.success(f"Synchronized: {res}")
        except Exception as exc:
            st.sidebar.error(f"Sync error: {exc}")
    else:
        st.sidebar.warning("Workspaces not configured or missing.")

st.sidebar.divider()
st.sidebar.header("🔑 LLM Configuration")
env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
input_key = st.sidebar.text_input(
    "Gemini API Key:",
    value=st.session_state.get("gemini_api_key", env_key),
    type="password",
    key="sidebar_gemini_key",
)
gemini_model = st.sidebar.selectbox(
    "Gemini Model:",
    ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"],
    index=0,
    key="sidebar_gemini_model",
)
st.session_state["gemini_model"] = gemini_model
if input_key:
    st.session_state["gemini_api_key"] = input_key


# ----------------------------------------------------------------------
# Live Background Headless Audit Monitor Fragment (Non-Blocking!)
# ----------------------------------------------------------------------
def _render_audit_jobs_monitor() -> None:
    jobs = list_audit_jobs(state_dir, limit=5)
    running_ids = {j["msg_id"] for j in jobs if j.get("status") == "RUNNING"}
    prev_running: set[str] = set(st.session_state.get("watched_running_audit_ids", []))
    st.session_state["watched_running_audit_ids"] = list(running_ids)

    # If a previously running background audit just finished, refresh full app so Thread Timeline shows verdict!
    if prev_running and not (prev_running <= running_ids) and not _IS_APPTEST:
        st.rerun(scope="app")

    if not jobs:
        return

    active_jobs = [j for j in jobs if j.get("status") == "RUNNING"]
    recent_done = [j for j in jobs if j.get("status") != "RUNNING"][:1]

    for job in active_jobs:
        elapsed_s = max(0, int(time.time() - float(job.get("started_epoch", time.time()))))
        revs = ", ".join(job.get("reviewers", []))
        with st.container(border=True):
            c_info, c_stop = st.columns([3.2, 1])
            with c_info:
                st.info(
                    f"🔄 **Headless Audit Running** — `{job['thread_id']}` (`{job['msg_id']}`) · "
                    f"Reviewer(s): **`{revs}`** · PID `{job.get('pid')}` · Elapsed: **`{elapsed_s}s`**"
                )
            with c_stop:
                if st.button("⏹️ Stop Audit", key=f"stop_audit_{job['msg_id']}", use_container_width=True):
                    stop_audit_job(state_dir, job["msg_id"])
                    st.rerun()
            tail = job.get("log_tail", "").strip()
            st.code(tail or "Mounting clean /tmp Git worktree and starting read-only auditor...", language="bash")

    if not active_jobs and recent_done:
        last_j = recent_done[0]
        status_lbl = last_j.get("status", "DONE")
        res_list = last_j.get("results") or []
        verdict_parts = [f"{r.get('reviewer')}: {r.get('verdict') or r.get('error')}" for r in res_list]
        summary_str = " | ".join(verdict_parts) if verdict_parts else status_lbl
        with st.expander(f"✅ Latest Headless Audit `{last_j['thread_id']}` (`{last_j['msg_id']}`): {summary_str}", expanded=False):
            if last_j.get("log_tail"):
                st.code(last_j["log_tail"], language="bash")


if not _IS_APPTEST and any(j.get("status") == "RUNNING" for j in list_audit_jobs(state_dir, limit=5)):
    render_live_audit_monitor = st.fragment(run_every=2.0)(_render_audit_jobs_monitor)
else:
    render_live_audit_monitor = _render_audit_jobs_monitor


# ----------------------------------------------------------------------
# 5 Primary Control Tower Tabs
# ----------------------------------------------------------------------
tab_threads, tab_knowledge, tab_scorecard, tab_proposals, tab_worker = st.tabs([
    "💬 Threads & Audit Cockpit",
    "📚 Shared Knowledge Explorer",
    "📊 10-Metric Scorecards",
    "📋 Proposals & Ideation",
    "🖥️ Live Worker & Eval Leases",
])

# ======================================================================
# TAB 1: Threads & Explicit Audit Dispatch Cockpit
# ======================================================================
with tab_threads:
    left_dispatch, right_threads = st.columns([1, 1.45])

    with left_dispatch:
        with st.container(border=True):
            st.subheader("📨 Explicit Audit & Message Dispatch")
            st.caption(
                "Target reviewers must always be selected explicitly. "
                "Read-only audits mount a clean `/tmp` Git worktree of the selected Code Checkout."
            )

            # Display flash confirmation banner from immediate post-dispatch rerun
            flash_notice = st.session_state.pop("last_dispatch_notice", None)
            if flash_notice:
                st.success(flash_notice["text"])
                for adv_item in flash_notice.get("advisories", []):
                    st.warning(adv_item)

            from_c1, code_c2 = st.columns(2)
            d_sender = from_c1.selectbox(
                "Requested By (`from_agent`):",
                ["human", "dev", "claude", "chatgpt", "main", "staging"],
                index=0,
                key="dispatch_from_agent",
            )
            d_target_checkout = code_c2.selectbox(
                "Code Checkout to Audit / Mount:",
                ["main", "dev", "claude", "chatgpt", "staging", "none"],
                index=0,
                key="dispatch_target_checkout",
            )
            if d_target_checkout != "none" and d_target_checkout in fleet_by_name:
                t_fs = fleet_by_name[d_target_checkout]
                st.caption(
                    f"📂 Mounts `{d_target_checkout}` (`{t_fs.workspace_path}`) · "
                    f"HEAD `{t_fs.head_sha[:7]}`: *{t_fs.head_subject[:65]}*"
                )

            d_intent = st.selectbox(
                "Intent:",
                [
                    Intent.AUDIT_REQUEST.value,
                    Intent.PROPOSAL_RFC.value,
                    Intent.REMEDIATION_READY.value,
                    Intent.INTEGRATION_REQUEST.value,
                    Intent.ADVISORY_NOTE.value,
                    Intent.ESCALATE_HUMAN.value,
                ],
                index=0,
                key="dispatch_intent",
            )
            d_subject = st.text_input(
                "Subject:",
                placeholder="e.g. Audit latest change on main branch on quantile strategy",
                key="dispatch_subject",
            )
            d_body = st.text_area(
                "Audit Request / Message Body:",
                placeholder="Describe commit scope, invariants to check, and validation evidence...",
                height=100,
                key="dispatch_body",
            )
            sha_c1, sha_c2 = st.columns(2)
            d_head_sha = sha_c1.text_input(
                "Commit SHA (`head_sha`):",
                placeholder="Blank = HEAD of selected checkout",
                key="dispatch_head_sha",
            )
            d_base_sha = sha_c2.text_input(
                "Base SHA (`base_sha`):",
                placeholder="Blank = HEAD~1 (latest commit diff)",
                key="dispatch_base_sha",
            )

            st.markdown("**Select Explicit Target Agent(s) (`to_agents`):**")
            chk_cols = st.columns(5)
            target_candidates = ["claude", "chatgpt", "dev", "main", "staging"]
            selected_targets: List[str] = []
            for i, cand in enumerate(target_candidates):
                if chk_cols[i].checkbox(cand, value=False, key=f"target_chk_{cand}"):
                    selected_targets.append(cand)

            default_mode_idx = 2 if Intent(d_intent) in READ_ONLY_AUTO_INTENTS else 0
            d_mode = st.radio(
                "Delivery & Execution Mode:",
                [
                    "Session Inbox Only",
                    "Queue Auto-Audit",
                    "Headless Read-Only Audit Now",
                ],
                index=default_mode_idx,
                key="dispatch_mode",
            )

            if st.button("🚀 Dispatch Message", type="primary", use_container_width=True, key="dispatch_submit_btn"):
                intent_obj = Intent(d_intent)
                if intent_obj in EXPLICIT_TARGET_INTENTS and not selected_targets:
                    st.error(
                        f"Explicit reviewer selection is required for `{d_intent}`: "
                        "please select at least one target agent before dispatching."
                    )
                elif not d_subject.strip():
                    st.error("Subject is required before dispatching a message.")
                elif d_mode == "Headless Read-Only Audit Now" and intent_obj not in READ_ONLY_AUTO_INTENTS:
                    st.error(
                        f"Headless auto-execution is forbidden for mutating intent `{d_intent}`. "
                        "Choose 'Session Inbox Only'."
                    )
                else:
                    try:
                        eff_head = d_head_sha.strip() or None
                        eff_base = d_base_sha.strip() or None

                        # Auto-resolve HEAD / HEAD~1 on the selected Code Checkout when auditing code!
                        if (
                            eff_head is None
                            and d_target_checkout != "none"
                            and intent_obj in (Intent.AUDIT_REQUEST, Intent.REMEDIATION_READY)
                        ):
                            target_ws = bus_svc.git_bridge.get_workspace_path(d_target_checkout)
                            if target_ws and target_ws.exists():
                                eff_head = "HEAD"
                                if eff_base is None:
                                    try:
                                        bus_svc.git_bridge.resolve_commit(target_ws, "HEAD~1")
                                        eff_base = "HEAD~1"
                                    except Exception:
                                        eff_base = None

                        res = bus_svc.send_verified(
                            from_agent=d_sender,
                            to_agents=selected_targets,
                            intent=intent_obj,
                            subject=d_subject.strip(),
                            body=d_body.strip(),
                            head_sha=eff_head,
                            base_sha=eff_base,
                            target_checkout=d_target_checkout,
                            require_clean_handoff=False,
                            auto_audit=(d_mode != "Session Inbox Only" and intent_obj in READ_ONLY_AUTO_INTENTS),
                            run_auto_audit_now=False,
                        )
                        env = res["envelope"]
                        msg_id = env["msg_id"]
                        tid = env["thread_id"]
                        to_names = [a.value if hasattr(a, "value") else str(a) for a in env["to_agents"]]
                        st.session_state["focus_thread_id"] = tid

                        if d_mode == "Headless Read-Only Audit Now":
                            job_meta = launch_background_audit(
                                msg_id=msg_id,
                                thread_id=tid,
                                reviewers=selected_targets,
                                root_dir=ROOT_DIR,
                                state_dir=state_dir,
                                config_path=cfg_path,
                                db_path=override_db,
                            )
                            notice_txt = (
                                f"Dispatched `{msg_id}` on thread `{tid}` -> `{to_names}`! "
                                f"Headless read-only audit launched in background (PID `{job_meta['pid']}`) — "
                                f"watch live progress on the right."
                            )
                        else:
                            notice_txt = (
                                f"Dispatched `{msg_id}` on thread `{tid}` -> `{to_names}` (mode: `{d_mode}`)!"
                            )

                        st.session_state["last_dispatch_notice"] = {
                            "text": notice_txt,
                            "advisories": env.get("advisories", []),
                        }
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Dispatch error: {exc}")

    with right_threads:
        st.subheader("🧵 Thread Timeline & Live Audit Monitor")
        render_live_audit_monitor()

        status_choices = ["ALL"] + [s.value for s in ThreadStatus]
        sel_status = st.selectbox("Filter by Thread Status:", status_choices, index=0, key="thread_status_filter")
        threads = bus_svc.store.list_threads(
            status=ThreadStatus(sel_status) if sel_status != "ALL" else None,
            limit=100,
        )

        if not threads:
            st.info("No threads recorded yet. Use the Dispatch panel on the left to start an audit or RFC thread.")
        else:
            thr_labels = [
                f"{t.thread_id} [{t.status.value}] R{t.latest_round} — {t.title} (by {t.created_by.value})"
                for t in threads
            ]
            thr_map = dict(zip(thr_labels, [t.thread_id for t in threads]))
            focus_tid = st.session_state.pop("focus_thread_id", None)
            default_idx = 0
            if focus_tid:
                for idx_t, t_obj in enumerate(threads):
                    if t_obj.thread_id == focus_tid:
                        default_idx = idx_t
                        st.session_state["thread_select"] = thr_labels[idx_t]
                        break

            sel_thr_lbl = st.selectbox("Select Thread:", thr_labels, index=default_idx, key="thread_select")
            bundle = bus_svc.store.get_thread(thr_map[sel_thr_lbl])

            if bundle:
                t_rec = bundle["thread"]
                t_msgs = bundle["messages"]
                t_findings = bundle["findings"]

                hdr_c1, hdr_c2, hdr_c3 = st.columns([2.0, 1.0, 1.0])
                with hdr_c1:
                    st.markdown(
                        f"**`{t_rec.thread_id}`**: {t_rec.title}  \n"
                        f"Status: **`{t_rec.status.value}`** · Round: `{t_rec.latest_round}` · "
                        f"Mirror: `{bundle['mirror_path']}`"
                    )
                with hdr_c2:
                    latest_auditable = next(
                        (m for m in reversed(t_msgs) if m.intent in READ_ONLY_AUTO_INTENTS),
                        None,
                    )
                    if latest_auditable and st.button(
                        "🤖 Re-Run Headless Audit",
                        use_container_width=True,
                        key="rerun_thread_audit_btn",
                    ):
                        revs = [a.value for a in latest_auditable.to_agents if a != AgentId.HUMAN]
                        if revs:
                            launch_background_audit(
                                msg_id=latest_auditable.msg_id,
                                thread_id=t_rec.thread_id,
                                reviewers=revs,
                                root_dir=ROOT_DIR,
                                state_dir=state_dir,
                                config_path=cfg_path,
                                db_path=override_db,
                            )
                            st.rerun()
                with hdr_c3:
                    if st.button(
                        "📦 Route to Staging for Full A/A",
                        use_container_width=True,
                        key="route_to_staging_btn",
                    ):
                        latest_ctx = next(
                            (m.code_context for m in reversed(t_msgs) if m.code_context is not None),
                            None,
                        )
                        bus_svc.store.send_message(
                            from_agent=AgentId.HUMAN,
                            to_agents=[AgentId.STAGING],
                            intent=Intent.INTEGRATION_REQUEST,
                            thread_id=t_rec.thread_id,
                            subject=f"Staging Integration & Full A/A: {t_rec.title}",
                            body="Approved for staging integration and full walk-forward A/A + A/B verification.",
                            code_context=latest_ctx,
                            auto_audit=False,
                        )
                        st.success(f"Routed `{t_rec.thread_id}` to `staging` session inbox (`IN_STAGING`).")

                if t_findings:
                    st.markdown("##### 🔍 Structured Findings Checklist (`P0` / `P1` / `P2`)")
                    f_df = pd.DataFrame(
                        [
                            {
                                "ID": f.id,
                                "Severity": f.severity.value,
                                "Status": f.status.value,
                                "Author": f.author_agent.value,
                                "Location": f"{f.file}:{f.lines}" if f.file else "—",
                                "Problem": f.problem,
                                "Suggested Fix": f.suggested_fix,
                                "Resolved SHA": f.resolved_in_sha or "—",
                                "Verified By": f.verified_by.value if f.verified_by else "—",
                            }
                            for f in t_findings
                        ]
                    )
                    st.dataframe(f_df, use_container_width=True, hide_index=True)

                    fc1, fc2, fc3 = st.columns([1.4, 1.4, 1])
                    sel_fid = fc1.selectbox("Finding ID:", [f.id for f in t_findings], key="finding_select")
                    sel_fstatus = fc2.selectbox(
                        "New Status:",
                        [s.value for s in FindingStatus],
                        key="finding_new_status",
                    )
                    if fc3.button("Update Finding", key="finding_update_btn", use_container_width=True):
                        bus_svc.store.update_finding_status(
                            sel_fid,
                            FindingStatus(sel_fstatus),
                            verified_by=AgentId.HUMAN if sel_fstatus == FindingStatus.VERIFIED.value else None,
                        )
                        st.success(f"Updated `{sel_fid}` -> `{sel_fstatus}`.")

                st.markdown("##### 📜 Chronological Message Timeline")
                for m in t_msgs:
                    to_txt = ", ".join(a.value for a in m.to_agents) or "broadcast"
                    verdict_badge = f" · Verdict: **`{m.verdict.value}`**" if m.verdict else ""
                    with st.container(border=True):
                        st.markdown(
                            f"**`{m.msg_id}`** (R{m.round_num}) `{m.from_agent.value}` ➡️ `{to_txt}` "
                            f"[`{m.intent.value}`]{verdict_badge}"
                        )
                        st.caption(f"{m.created_at} — **{m.subject}**")
                        for adv in m.advisories:
                            st.warning(adv)
                        if m.body:
                            st.markdown(m.body)
                        if m.code_context and m.code_context.head_sha:
                            with st.expander(
                                f"🔎 Commit Context: `{m.code_context.branch}` @ `{m.code_context.head_sha[:8]}` "
                                f"({len(m.code_context.changed_files)} file(s))"
                            ):
                                try:
                                    diff_data = bus_svc.git_bridge.inspect_commit(
                                        from_agent=m.from_agent if m.from_agent != AgentId.HUMAN else bus_svc.git_bridge.golden_agent,
                                        thread_id=m.thread_id,
                                        head_sha=m.code_context.head_sha,
                                        base_sha=m.code_context.base_sha,
                                        source_workspace=m.code_context.source_workspace,
                                        max_diff_chars=8000,
                                    )
                                    st.code(diff_data["diff_patch"] or diff_data["diff_stat"], language="diff")
                                except Exception as exc:
                                    st.caption(f"Diff preview unavailable: {exc}")

                with st.expander("💬 Post Human / Agent Reply on This Thread"):
                    r_from = st.selectbox(
                        "Reply As:",
                        ["human", "dev", "claude", "chatgpt", "staging", "main"],
                        key="reply_from_agent",
                    )
                    r_to = st.multiselect(
                        "Reply To (`to_agents`):",
                        ["dev", "claude", "chatgpt", "staging", "main", "human"],
                        default=[t_rec.created_by.value],
                        key="reply_to_agents",
                    )
                    r_intent = st.selectbox(
                        "Reply Intent:",
                        [
                            Intent.ADVISORY_NOTE.value,
                            Intent.AUDIT_VERDICT.value,
                            Intent.PEER_CRITIQUE.value,
                            Intent.REMEDIATION_READY.value,
                            Intent.STAGING_RESULT.value,
                            Intent.ESCALATE_HUMAN.value,
                        ],
                        key="reply_intent",
                    )
                    r_verdict = st.selectbox(
                        "Verdict (optional):",
                        ["NONE"] + [v.value for v in AuditVerdictType],
                        key="reply_verdict",
                    )
                    r_body = st.text_area("Reply Body:", key="reply_body")
                    if st.button("Send Reply", key="reply_submit_btn"):
                        if Intent(r_intent) in EXPLICIT_TARGET_INTENTS and not r_to:
                            st.error("Explicit `to_agents` selection is required for this reply intent.")
                        else:
                            bus_svc.store.send_message(
                                from_agent=r_from,
                                to_agents=r_to,
                                intent=Intent(r_intent),
                                thread_id=t_rec.thread_id,
                                subject=f"Re: {t_rec.title}",
                                body=r_body,
                                verdict=AuditVerdictType(r_verdict) if r_verdict != "NONE" else None,
                                auto_audit=False,
                            )
                            st.success("Reply posted to thread!")
                            st.rerun()

# ======================================================================
# TAB 2: Shared Cross-Checkout Knowledge & Research Explorer
# ======================================================================
with tab_knowledge:
    st.subheader("📚 Cross-Checkout Institutional Memory (`research_notes/`, `reports/`, & A/B Archive)")
    kc1, kc2, kc3, kc4 = st.columns([2.2, 1, 1, 1])
    k_query = kc1.text_input(
        "FTS5 Search Query:",
        placeholder="e.g. Stage-1 path ranker, volatility, lookahead, focal loss...",
        key="know_search_query",
    )
    k_cat = kc2.selectbox(
        "Category:",
        ["ALL", "research_note", "experiment_report", "ab_negative", "ab_positive", "ab_neutral"],
        key="know_cat_filter",
    )
    k_agent = kc3.selectbox(
        "Source Checkout:",
        ["ALL", "claude", "dev", "chatgpt", "staging", "main"],
        key="know_agent_filter",
    )
    if kc4.button("🔄 Sync All 5 Checkouts", key="know_sync_btn", use_container_width=True):
        sync_stats = bus_svc.knowledge.sync_all_workspaces()
        st.success(
            f"Synced 5 checkouts: scanned={sync_stats['scanned']}, "
            f"added={sync_stats['added']}, deduped={sync_stats['deduped']}"
        )

    with st.expander("➕ Publish Concluded / Negative A/B Experiment to Institutional Memory"):
        pc1, pc2, pc3 = st.columns(3)
        p_agent = pc1.selectbox("Source Agent:", ["dev", "claude", "chatgpt", "staging", "main"], key="pub_agent")
        p_outcome = pc2.selectbox("Outcome:", ["NEGATIVE", "POSITIVE", "NEUTRAL"], key="pub_outcome")
        p_strat = pc3.selectbox("Strategy:", [s.value for s in StrategyTarget], key="pub_strategy")
        p_title = st.text_input("Experiment Title:", key="pub_title")
        p_summary = st.text_area("Empirical Findings Summary:", key="pub_summary")
        p_fail = st.text_input("Failure Mode / Conditions Not To Repeat:", key="pub_failures")
        if st.button("Publish Experiment Record", key="pub_submit_btn"):
            if not p_title.strip() or not p_summary.strip():
                st.error("Title and Summary are required to publish an experiment record.")
            else:
                entry = bus_svc.knowledge.publish_experiment_record(
                    source_agent=p_agent,
                    title=p_title.strip(),
                    summary=p_summary.strip(),
                    outcome=p_outcome,
                    strategy=p_strat,
                    failure_conditions=p_fail.strip(),
                )
                st.success(f"Published `{entry.entry_id}` (`{entry.category}`) and broadcast to peers!")

    hits = bus_svc.store.search_knowledge(
        query=k_query,
        category=None if k_cat == "ALL" else k_cat,
        source_agent=None if k_agent == "ALL" else k_agent,
        limit=40,
    )
    st.caption(f"Showing {len(hits)} indexed knowledge artifact(s).")
    for item in hits:
        with st.container(border=True):
            outcome_str = f" · Outcome: **`{item.outcome}`**" if item.outcome else ""
            st.markdown(
                f"**`{item.entry_id}`** — **{item.title}**  \n"
                f"Source: `{item.source_agent.value}` (`{item.source_branch}`) · "
                f"Category: `{item.category}`{outcome_str}"
            )
            st.write(item.summary)
            if item.artifact_path:
                st.caption(f"Artifact: `{item.artifact_path}`")
            if item.body_excerpt:
                with st.expander("Read Excerpt"):
                    st.markdown(item.body_excerpt[:3000])

# ======================================================================
# TAB 3: 10-Metric Institutional A/B & Staged Scorecards
# ======================================================================
with tab_scorecard:
    st.subheader("📊 10-Metric Walk-Forward A/B & Staging Scorecards")
    sc_mode = st.radio(
        "Scorecard Source:",
        ["Direct CSV File / Staged Run", "Evaluated Proposals"],
        horizontal=True,
        key="scorecard_source_mode",
    )

    if sc_mode == "Direct CSV File / Staged Run":
        discovered_csvs: List[Path] = []
        if results_dir.exists():
            discovered_csvs = sorted(
                [p for p in results_dir.glob("*.csv") if "_ab-" in p.name or "ab" in p.name.lower()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:40]

        default_csv = str(discovered_csvs[0]) if discovered_csvs else ""
        if discovered_csvs:
            sel_disc = st.selectbox(
                "Recent A/B Result CSVs:",
                [str(p) for p in discovered_csvs],
                index=0,
                key="scorecard_csv_select",
            )
            default_csv = sel_disc

        custom_csv = st.text_input(
            "Inspect Specific CSV Path (or comma-separated DEV,RECENT,BENCHMARK CSVs for Staged TOTAL):",
            value=default_csv,
            key="scorecard_csv_path",
        )

        if custom_csv.strip():
            raw_paths = [Path(tok.strip()) for tok in custom_csv.split(",") if tok.strip()]
            try:
                if len(raw_paths) > 1:
                    staged_map = CSVParser.parse_staged_run(raw_paths)
                    stage_names = list(staged_map.keys())
                    sel_stage = st.radio("Stage Split:", stage_names, horizontal=True, key="scorecard_stage_radio")
                    active_sc = staged_map[sel_stage]
                else:
                    active_sc = CSVParser.parse_result_file(raw_paths[0])

                st.markdown(
                    f"#### `{active_sc.strategy}` vs `{active_sc.variant}` — Stage `{active_sc.fold_set}` "
                    f"({active_sc.num_folds} folds, commit `{active_sc.git_commit}`)"
                )
                st.dataframe(build_ten_metric_dataframe(active_sc), use_container_width=True, hide_index=True)

                if active_sc.regime_deltas:
                    st.markdown("##### Regime Return Deltas")
                    r_cols = st.columns(len(active_sc.regime_deltas))
                    for i, (reg, val) in enumerate(active_sc.regime_deltas.items()):
                        r_cols[i].metric(reg, f"{val:+.2f}%")
            except InvalidScorecardCSVError as inv_err:
                st.error(f"Invalid A/B Scorecard CSV: {inv_err}")
            except FileNotFoundError as fnf_err:
                st.error(f"CSV file not found: {fnf_err}")
            except Exception as exc:
                st.error(f"Failed to parse scorecard CSV: {exc}")
        else:
            st.info("Provide or select an A/B fold result CSV path to inspect its 10-metric scorecard.")
    else:
        all_props_sc = pm.list_proposals()
        eval_props = [p for p in all_props_sc if p.scorecard_summary is not None]
        if not eval_props:
            st.info("No evaluated proposals have scorecards attached yet.")
        else:
            score_options = {f"{p.id} - {p.metadata.title}": p.id for p in eval_props}
            sel_eval_key = st.selectbox("Select Evaluated Proposal:", list(score_options.keys()), key="prop_sc_select")
            s_prop = pm.load_proposal(score_options[sel_eval_key])
            if s_prop and s_prop.scorecard_summary:
                st.markdown(f"### {s_prop.id}: {s_prop.metadata.title}")
                st.dataframe(build_ten_metric_dataframe(s_prop.scorecard_summary), use_container_width=True, hide_index=True)
                if s_prop.status == ProposalStatus.EVALUATED and st.button(
                    "🏆 Promote Winning Proposal to Main Branch", type="primary", key="promote_prop_btn"
                ):
                    if git_mgr and golden_path:
                        try:
                            target_ws = ws_paths.get(s_prop.metadata.assigned_branch, golden_path)
                            new_sha = git_mgr.promote_to_main(
                                candidate_ws=target_ws,
                                golden_ws=golden_path,
                                proposal_id=s_prop.id,
                                title=s_prop.metadata.title,
                            )
                            pm.promote_proposal(s_prop)
                            st.success(f"Promoted `{s_prop.id}` to main (`{new_sha[:8]}`)!")
                        except Exception as exc:
                            st.error(f"Promotion failed: {exc}")

# ======================================================================
# TAB 4: Proposals Kanban, Review & Grounded Ideation
# ======================================================================
with tab_proposals:
    all_proposals = pm.list_proposals()
    sub_kanban, sub_review, sub_ideation = st.tabs([
        "📋 Proposal Kanban",
        "🔎 Review & Bus RFC Routing",
        "💡 Grounded Ideation Engine",
    ])

    with sub_kanban:
        col1, col2, col3, col4, col5 = st.columns(5)
        columns_map = {
            "Review": (col1, [ProposalStatus.DRAFT, ProposalStatus.HUMAN_REVIEW, ProposalStatus.PEER_REVIEW]),
            "Approved": (col2, [ProposalStatus.APPROVED]),
            "Running": (col3, [ProposalStatus.IMPLEMENTING, ProposalStatus.AB_TESTING]),
            "Evaluated": (col4, [ProposalStatus.EVALUATED]),
            "Done": (col5, [ProposalStatus.PROMOTED, ProposalStatus.REJECTED]),
        }
        for col_name, (ui_col, status_list) in columns_map.items():
            with ui_col:
                st.markdown(f"#### {col_name}")
                props_in_col = [p for p in all_proposals if p.status in status_list]
                if not props_in_col:
                    st.caption("None")
                for p in props_in_col:
                    with st.container(border=True):
                        st.markdown(f"**{p.id}** `{p.status.value}`")
                        st.markdown(f"*{p.metadata.title}*")
                        st.caption(f"Author: `{p.author_model.value}` · Branch: `{p.metadata.assigned_branch}`")

    with sub_review:
        if not all_proposals:
            st.info("No proposals available yet.")
        else:
            prop_options = {f"{p.id} - {p.metadata.title} ({p.status.value})": p.id for p in all_proposals}
            sel_p_key = st.selectbox("Select Proposal:", list(prop_options.keys()), key="prop_review_select")
            proposal = pm.load_proposal(prop_options[sel_p_key])
            if proposal:
                c_meta, c_body = st.columns([1, 2])
                with c_meta:
                    st.write(f"**ID**: `{proposal.id}` | **Status**: `{proposal.status.value}`")
                    st.write(f"**Strategy**: `{proposal.metadata.target_strategy.value}` | **Branch**: `{proposal.metadata.assigned_branch}`")
                    if proposal.status in (ProposalStatus.HUMAN_REVIEW, ProposalStatus.DRAFT, ProposalStatus.PEER_REVIEW):
                        if st.button("✅ Approve for Implementation", type="primary", use_container_width=True, key="prop_approve_btn"):
                            ProposalStateMachine.approve(proposal)
                            pm.save_proposal(proposal)
                            st.success(f"Approved `{proposal.id}`!")

                    st.markdown("**Send `PROPOSAL_RFC` to Explicit Bus Reviewers:**")
                    rfc_targets: List[str] = []
                    for cand in ("claude", "chatgpt", "dev"):
                        if st.checkbox(f"Peer: {cand}", value=False, key=f"prop_rfc_chk_{cand}"):
                            rfc_targets.append(cand)
                    if st.button("📨 Dispatch Proposal RFC on Bus", use_container_width=True, key="prop_send_bus_rfc_btn"):
                        if not rfc_targets:
                            st.error("Select at least one explicit reviewer before dispatching a Proposal RFC.")
                        else:
                            env_res = bus_svc.send_verified(
                                from_agent=proposal.author_model.value,
                                to_agents=rfc_targets,
                                intent=Intent.PROPOSAL_RFC,
                                subject=f"RFC {proposal.id}: {proposal.metadata.title}",
                                body=f"{proposal.hypothesis}\n\n{proposal.implementation_spec}",
                                proposal_id=proposal.id,
                                auto_audit=True,
                            )
                            st.success(f"Created Bus RFC thread `{env_res['envelope']['thread_id']}` -> `{rfc_targets}`!")

                with c_body:
                    st.markdown("#### Hypothesis")
                    st.markdown(proposal.hypothesis)
                    if proposal.implementation_spec:
                        st.markdown("#### Implementation Spec")
                        st.markdown(proposal.implementation_spec)

    with sub_ideation:
        f_author = st.selectbox(
            "Author Model:",
            [ModelName.GEMINI.value, ModelName.CLAUDE.value, ModelName.CHATGPT.value],
            key="gen_author",
        )
        f_strategy = st.selectbox("Target Strategy:", [s.value for s in StrategyTarget], key="gen_strat")
        f_theme = st.selectbox(
            "Research Theme:",
            [
                "Volatility scaling & normalization (Parkinson / Garman-Klass)",
                "Dynamic regime filtering (Bull/Bear x Hi/Lo vol)",
                "Stop-loss & profit-target barrier calibration",
                "Quantile ranking feature expansion & volume signals",
                "Overfitting & lookahead audit fix",
            ],
            key="gen_theme",
        )
        if st.button("🚀 Generate Grounded Proposal RFC", type="primary", key="gen_submit_btn"):
            new_prop = IdeationEngine.generate_proposal(
                author_model=ModelName(f_author),
                strategy=StrategyTarget(f_strategy),
                theme=f_theme,
                results_dir=results_dir,
                rejected_dir=pm.rejected_dir,
                next_id=pm.get_next_proposal_id(),
                api_key=st.session_state.get("gemini_api_key"),
                model_name=st.session_state.get("gemini_model", "gemini-3.8-flash"),
            )
            ProposalStateMachine.submit_to_human(new_prop)
            pm.save_proposal(new_prop)
            st.success(f"Generated proposal `{new_prop.id}`: {new_prop.metadata.title}")

# ======================================================================
# TAB 5: Live Auto-Audit Worker, Host Eval Leases & Proposal Runner
# ======================================================================
with tab_worker:
    w_col1, w_col2 = st.columns(2)

    with w_col1:
        with st.container(border=True):
            st.subheader("🤖 Headless Read-Only Auto-Audit Queue")
            st.caption(
                "Polls un-dispatched `AUDIT_REQUEST`, `PROPOSAL_RFC`, and `REMEDIATION_READY` messages "
                "and launches them in isolated `/tmp/rl-worktrees/` background sessions."
            )
            if st.button("▶️ Launch Pending Read-Only Auto-Audits", type="primary", key="run_pending_audits_btn"):
                claimed = bus_svc.store.claim_pending_auto_audits(limit=5)
                if not claimed:
                    st.info("No pending auto-audit messages in queue.")
                else:
                    for c_msg in claimed:
                        revs = [a.value for a in c_msg.to_agents if a != AgentId.HUMAN]
                        launch_background_audit(
                            msg_id=c_msg.msg_id,
                            thread_id=c_msg.thread_id,
                            reviewers=revs,
                            root_dir=ROOT_DIR,
                            state_dir=state_dir,
                            config_path=cfg_path,
                            db_path=override_db,
                        )
                    st.success(f"Launched {len(claimed)} background headless audit job(s)!")
                    st.rerun()

            for j_item in list_audit_jobs(state_dir, limit=6):
                st.caption(
                    f"`{j_item['msg_id']}` ({j_item['thread_id']}) · `{j_item.get('status')}` · "
                    f"PID `{j_item.get('pid')}` · Reviewers `{j_item.get('reviewers')}`"
                )

        with st.container(border=True):
            st.subheader("🔒 Active Host Evaluation Leases")
            if not active_leases:
                st.caption("No active evaluation leases currently held.")
            else:
                st.dataframe(pd.DataFrame(active_leases), use_container_width=True, hide_index=True)
                sel_lid = st.selectbox(
                    "Select Lease to Release:",
                    [ls["lease_id"] for ls in active_leases],
                    key="lease_release_select",
                )
                if st.button("🔓 Release Selected Lease", key="lease_release_btn"):
                    bus_svc.capacity.release_eval_lease(sel_lid)
                    st.success(f"Released lease `{sel_lid}`.")

    with w_col2:
        with st.container(border=True):
            st.subheader("🖥️ In-Process Approved Proposal Execution")
            ready_props = [p for p in pm.list_proposals() if p.status == ProposalStatus.APPROVED]
            if not ready_props:
                st.info("No proposals currently in `APPROVED` status.")
            else:
                exec_options = {
                    f"{p.id} - {p.metadata.title} ({p.metadata.assigned_branch})": p.id for p in ready_props
                }
                sel_exec_key = st.selectbox("Approved Proposal:", list(exec_options.keys()), key="exec_prop_select")
                target_exec_prop = pm.load_proposal(exec_options[sel_exec_key])
                if target_exec_prop and st.button(
                    "🚀 Start In-Process Agent Execution & Backtest",
                    type="primary",
                    key="exec_prop_run_btn",
                ):
                    log_box = st.empty()
                    lines: List[str] = []

                    def _on_stream(ln: str) -> None:
                        lines.append(ln)
                        log_box.code("\n".join(lines[-40:]), language="bash")

                    res, sc = dispatcher.run_proposal_execution(
                        proposal=target_exec_prop,
                        on_output=_on_stream,
                        auto_eval=True,
                    )
                    if res.success:
                        st.success(f"Execution succeeded! Commit: `{res.commit_sha}`")
                        if sc:
                            st.success(f"Backtest complete! Delta PnL: `{sc.delta_total_pnl:+.2f}%`")
                    else:
                        st.error(f"Execution failed: {res.error}")
