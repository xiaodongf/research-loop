"""Tier 4B Tests: Headless Streamlit AppTest verification of the 5-tab Control Tower."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import pytest
from streamlit.testing.v1 import AppTest

from src.bus.mcp_server import BusService
from src.bus.schema import AgentId, ThreadStatus

# Import fixture from bus conftest
from tests.bus.conftest import five_repo_topology  # noqa: F401

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DASHBOARD_PATH = ROOT_DIR / "src" / "ui" / "dashboard.py"


def test_dashboard_fleet_cards_explicit_reviewer_validation_and_staging_route(
    five_repo_topology: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path: Path = five_repo_topology["config_path"]
    monkeypatch.setenv("RL_CONFIG_PATH", str(cfg_path))
    monkeypatch.delenv("RL_DB_PATH", raising=False)

    at = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=20)
    at.run()
    assert not at.exception

    # 1. Verify advisory behind-main warning is rendered for chatgpt (3 behind main)
    warning_texts = [w.value for w in at.warning]
    assert any("3 behind main" in w for w in warning_texts)

    # 2. Decision #5 UI Gate: Clicking Dispatch with ZERO target checkboxes checked MUST show error & create 0 threads
    at.text_input(key="dispatch_subject").set_value("Audit Stage-1 ranker")
    at.text_area(key="dispatch_body").set_value("Check shift(1) causality")
    at.button(key="dispatch_submit_btn").click().run()
    assert not at.exception
    assert any("Explicit reviewer selection is required" in e.value for e in at.error)

    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])
    assert svc.store.list_threads() == []

    # 3. Check `claude` and `chatgpt` explicitly and dispatch -> creates THR-0001 cleanly!
    at.checkbox(key="target_chk_claude").check()
    at.checkbox(key="target_chk_chatgpt").check()
    at.button(key="dispatch_submit_btn").click().run()
    assert not at.exception
    assert any("THR-0001" in s.value for s in at.success)

    threads = svc.store.list_threads()
    assert len(threads) == 1
    assert threads[0].thread_id == "THR-0001"
    assert len(svc.store.get_inbox(AgentId.CLAUDE)) == 1
    assert len(svc.store.get_inbox(AgentId.CHATGPT)) == 1

    # 4. Re-run so thread selector picks up THR-0001, then click `Route to Staging for Full A/A`
    at.run()
    assert not at.exception
    at.button(key="route_to_staging_btn").click().run()
    assert not at.exception
    assert any("IN_STAGING" in s.value for s in at.success)

    bundle = svc.store.get_thread("THR-0001")
    assert bundle is not None
    assert bundle["thread"].status == ThreadStatus.IN_STAGING
    staging_inbox = svc.store.get_inbox(AgentId.STAGING)
    assert len(staging_inbox) == 1
    assert staging_inbox[0].intent.value == "INTEGRATION_REQUEST"


def test_dashboard_ten_metric_scorecard_and_malformed_csv_rejection(
    five_repo_topology: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path: Path = five_repo_topology["config_path"]
    results_dir: Path = five_repo_topology["results_dir"]
    monkeypatch.setenv("RL_CONFIG_PATH", str(cfg_path))

    good_csv = results_dir / "standalone_tb_ab-volgate_DEV_20261111_101500_7665697.csv"
    good_csv.write_text(
        "fold,regime,spy,base_n,base_win,base_total_return,base_max_dd,base_sharpe,base_prec1,base_avg_exp,"
        "var_n,var_win,var_total_return,var_max_dd,var_sharpe,var_prec1,var_avg_exp\n"
        "2025-01,BULL,3.0,25,0.52,7.5,-4.2,1.15,56.0,31.0,26,0.58,10.8,-3.5,1.42,61.0,33.0\n",
        encoding="utf-8",
    )

    bad_csv = results_dir / "all_cut_trades_dump.csv"
    bad_csv.write_text("symbol,entry_ts,pnl\nMSFT,2026-02-01,1.2\n", encoding="utf-8")

    at = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=20)
    at.run()
    assert not at.exception

    # Valid A/B CSV renders 10-metric dataframe with zero scorecard errors
    at.text_input(key="scorecard_csv_path").set_value(str(good_csv)).run()
    assert not at.exception
    assert not any("Invalid A/B Scorecard CSV" in e.value for e in at.error)
    assert any(len(df.value) == 10 for df in at.dataframe)

    # Malformed / non-A/B CSV surfaces InvalidScorecardCSVError cleanly without crashing UI
    at.text_input(key="scorecard_csv_path").set_value(str(bad_csv)).run()
    assert not at.exception
    assert any("Invalid A/B Scorecard CSV" in e.value for e in at.error)


def test_dashboard_headless_audit_dispatch_auto_resolves_main_head_and_launches_nonblocking(
    five_repo_topology: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Regression test for 'after I click dispatch, nothing happened':
    Leaving `Commit SHA` blank while dispatching `Headless Read-Only Audit Now` must:
    1) Auto-resolve `HEAD` on the selected Code Checkout (`main`),
    2) Spawn `launch_background_audit` non-blockingly and immediately show a confirmation banner on rerun.
    """
    cfg_path: Path = five_repo_topology["config_path"]
    monkeypatch.setenv("RL_CONFIG_PATH", str(cfg_path))
    monkeypatch.delenv("RL_DB_PATH", raising=False)

    launched_calls: list[Dict[str, Any]] = []

    def _fake_launch_background_audit(**kwargs: Any) -> Dict[str, Any]:
        launched_calls.append(kwargs)
        return {
            "msg_id": kwargs["msg_id"],
            "thread_id": kwargs["thread_id"],
            "reviewers": kwargs["reviewers"],
            "pid": 424242,
            "status": "RUNNING",
        }

    monkeypatch.setattr("src.bus.audit_worker.launch_background_audit", _fake_launch_background_audit)

    at = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=20)
    at.run()
    assert not at.exception

    at.text_input(key="dispatch_subject").set_value(
        "do an audit of the latest change on main branch on quantile strategy"
    )
    at.text_area(key="dispatch_body").set_value(
        "do a comprehensive audit of the latest change on main branch on quantile strategy"
    )
    # Leave dispatch_head_sha blank on purpose!
    at.checkbox(key="target_chk_chatgpt").check()
    at.radio(key="dispatch_mode").set_value("Headless Read-Only Audit Now")
    at.button(key="dispatch_submit_btn").click().run()
    assert not at.exception

    # Immediate confirmation banner rendered after non-blocking st.rerun()
    assert any("launched in background" in s.value and "424242" in s.value for s in at.success)
    assert len(launched_calls) == 1
    assert launched_calls[0]["thread_id"] == "THR-0001"
    assert launched_calls[0]["reviewers"] == ["chatgpt"]

    svc = BusService(root_dir=five_repo_topology["hub_root"], config=five_repo_topology["config"])
    msg = svc.store.get_message("MSG-000001")
    assert msg is not None
    assert msg.code_context is not None
    assert msg.code_context.head_sha == five_repo_topology["main_head"]
    assert msg.code_context.branch == "main"
