"""Research Loop: Multi-Model Trading Research Control Tower (Streamlit Web Dashboard)."""
import os, sys, yaml, time
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.schema import (
    ModelName, StrategyTarget, ProposalStatus, ReviewVerdict, Proposal
)
from src.core.proposal_manager import ProposalManager
from src.core.state_machine import ProposalStateMachine
from src.core.ideation import ContextBuilder, PromptBuilder
from src.core.git_manager import GitManager
from src.agents.dispatcher import AgentDispatcher
from src.harness.eval_bridge import EvalBridge


def load_config() -> dict:
    conf_path = ROOT_DIR / "config.yaml"
    if conf_path.exists():
        with open(conf_path) as f:
            return yaml.safe_load(f)
    return {}


# Page config
st.set_page_config(
    page_title="Research Loop | Control Tower",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

config = load_config()
pm = ProposalManager(root_dir=ROOT_DIR)

# Workspaces & Git manager setup
workspaces_conf = config.get("workspaces", {})
ws_paths = {k: Path(v["path"]) for k, v in workspaces_conf.items() if "path" in v and Path(v["path"]).exists()}
golden_path = Path(workspaces_conf.get("gemini", {}).get("path", "")) if "gemini" in workspaces_conf else None
git_mgr = GitManager(workspaces=ws_paths, golden_branch=config.get("golden_truth", {}).get("branch", "main")) if ws_paths else None

eval_conf = config.get("evaluation", {})
eval_bridge = EvalBridge(
    python_bin=Path(eval_conf.get("python_bin", sys.executable)),
    results_dir=Path(eval_conf.get("results_dir", ROOT_DIR / "evaluation" / "outputs" / "results")),
)
dispatcher = AgentDispatcher(config=config, proposal_manager=pm, eval_bridge=eval_bridge)

# Top Header
st.title("🏛️ Research Loop: Multi-Model Research Control Tower")
st.caption("Autonomous Discovery, Peer Review, and Walk-Forward Backtesting across Gemini, Claude Code, and ChatGPT.")

# Sidebar Workspace Status
st.sidebar.header("🌿 Branch Workspaces")
for key, conf in workspaces_conf.items():
    ws_p = Path(conf.get("path", ""))
    branch = conf.get("branch", "")
    is_clean = git_mgr.is_clean(ws_p) if git_mgr and ws_p.exists() else False
    head = git_mgr.get_head_sha(ws_p)[:7] if git_mgr and ws_p.exists() and (ws_p / ".git").exists() else "N/A"
    status_icon = "🟢" if is_clean else "🟡 Dirty"
    st.sidebar.markdown(f"**{conf.get('name', key.title())}** (`{branch}`)")
    st.sidebar.caption(f"Commit: `{head}` | Status: {status_icon}")

st.sidebar.divider()
if st.sidebar.button("⚡ Rebase All Branches from Main", use_container_width=True):
    if git_mgr and golden_path and golden_path.exists():
        with st.spinner("Rebasing all model workspaces onto golden main..."):
            try:
                res = git_mgr.sync_all_workspaces(golden_ws=golden_path)
                st.sidebar.success(f"Synchronized: {res}")
            except Exception as e:
                st.sidebar.error(f"Sync error: {e}")
    else:
        st.sidebar.warning("Workspaces not configured or missing.")

# Navigation Tabs
tab_board, tab_cockpit, tab_generate, tab_scorecard, tab_terminal = st.tabs([
    "📋 Proposal Kanban",
    "🔎 Review & Routing Cockpit",
    "💡 Grounded Ideation",
    "📊 A/B Scorecard",
    "🖥️ Live Terminal Stream",
])

# -------------------------------------------------------------
# TAB 1: Proposal Kanban Board
# -------------------------------------------------------------
with tab_board:
    st.subheader("Proposal Pipeline")
    all_proposals = pm.list_proposals()

    col1, col2, col3, col4, col5 = st.columns(5)
    columns_map = {
        "Review": (col1, [ProposalStatus.DRAFT, ProposalStatus.HUMAN_REVIEW, ProposalStatus.PEER_REVIEW]),
        "Approved": (col2, [ProposalStatus.APPROVED]),
        "Running": (col3, [ProposalStatus.IMPLEMENTING, ProposalStatus.AB_TESTING]),
        "Evaluated": (col4, [ProposalStatus.EVALUATED]),
        "Promoted / Done": (col5, [ProposalStatus.PROMOTED, ProposalStatus.REJECTED]),
    }

    badge_colors = {"gemini": "🔵", "claude": "🟣", "chatgpt": "🟢"}

    for col_name, (ui_col, status_list) in columns_map.items():
        with ui_col:
            st.markdown(f"### {col_name}")
            props_in_col = [p for p in all_proposals if p.status in status_list]
            if not props_in_col:
                st.caption("None")
            for p in props_in_col:
                badge = badge_colors.get(p.author_model.value, "⚪")
                with st.container(border=True):
                    st.markdown(f"**{p.id}** {badge} `{p.status.value}`")
                    st.markdown(f"*{p.metadata.title}*")
                    st.caption(f"Strategy: `{p.metadata.target_strategy.value}` | Branch: `{p.metadata.assigned_branch}`")
                    if p.metadata.reviewers:
                        st.caption(f"Reviews: {len(p.metadata.reviewers)} recorded")

# -------------------------------------------------------------
# TAB 2: Review & Routing Cockpit
# -------------------------------------------------------------
with tab_cockpit:
    st.subheader("Human-in-the-Loop Review Cockpit")
    active_props = [p for p in all_proposals if p.status not in (ProposalStatus.PROMOTED, ProposalStatus.REJECTED)]
    
    if not all_proposals:
        st.info("No proposals available. Generate one in the Grounded Ideation tab!")
    else:
        prop_options = {f"{p.id} - {p.metadata.title} ({p.status.value})": p.id for p in all_proposals}
        selected_key = st.selectbox("Select Proposal for Review:", list(prop_options.keys()))
        selected_id = prop_options[selected_key]
        proposal = pm.load_proposal(selected_id)

        c_meta, c_content = st.columns([1, 2])
        with c_meta:
            st.markdown("#### Proposal Metadata")
            st.write(f"**ID**: {proposal.id}")
            st.write(f"**Author**: {proposal.author_model.value.title()}")
            st.write(f"**Status**: `{proposal.status.value}`")
            st.write(f"**Target Strategy**: `{proposal.metadata.target_strategy.value}`")
            st.write(f"**Assigned Branch**: `{proposal.metadata.assigned_branch}`")
            if proposal.proposed_files:
                st.markdown("**Proposed Files:**")
                for f in proposal.proposed_files:
                    st.markdown(f"- `{f}`")

            st.divider()
            st.markdown("#### Governance Actions")

            # Approve & Implement
            if proposal.status in (ProposalStatus.HUMAN_REVIEW, ProposalStatus.DRAFT):
                if st.button("✅ Approve & Implement", type="primary", use_container_width=True):
                    ProposalStateMachine.approve(proposal)
                    pm.save_proposal(proposal)
                    st.success(f"Proposal {proposal.id} APPROVED! Run in Terminal Stream tab.")
                    st.rerun()

            # Route for Peer Review
            st.markdown("##### Route to Model for Critique")
            route_model = st.selectbox("Select Reviewer Model:", [m.value for m in ModelName if m != proposal.author_model])
            if st.button(f"➡️ Route to {route_model.title()}", use_container_width=True):
                ProposalStateMachine.route_to_peer(proposal, reviewer=ModelName(route_model))
                pm.save_proposal(proposal)
                st.info(f"Routed {proposal.id} to {route_model} (status: PEER_REVIEW).")
                st.rerun()

            # Add Critique
            if proposal.status == ProposalStatus.PEER_REVIEW:
                st.markdown("##### Submit Peer Critique")
                critique_verdict = st.selectbox("Verdict:", [v.value for v in ReviewVerdict])
                critique_text = st.text_area("Critique / Audit Notes:", placeholder="Lookahead bias, shift(1), etc.")
                if st.button("Submit Peer Critique", use_container_width=True):
                    ProposalStateMachine.add_peer_critique(
                        proposal,
                        reviewer=ModelName(route_model),
                        verdict=ReviewVerdict(critique_verdict),
                        comment=critique_text,
                    )
                    pm.save_proposal(proposal)
                    st.success("Critique added! Returned to HUMAN_REVIEW.")
                    st.rerun()

            # Reject
            st.markdown("##### Reject Proposal")
            reject_reason = st.text_input("Rejection Reason:", "Failed theoretical criteria")
            if st.button("❌ Reject Proposal", use_container_width=True):
                pm.reject_proposal(proposal, reason=reject_reason)
                st.warning(f"Proposal {proposal.id} REJECTED and archived.")
                st.rerun()

        with c_content:
            st.markdown("#### Hypothesis")
            st.info(proposal.hypothesis)

            if proposal.implementation_spec:
                st.markdown("#### Implementation Spec")
                st.code(proposal.implementation_spec, language="markdown")

            st.markdown("#### Multi-Model Discussion & Audit Thread")
            if proposal.discussion_log:
                st.markdown(proposal.discussion_log)
            else:
                st.caption("No peer reviews or discussions recorded yet.")

# -------------------------------------------------------------
# TAB 3: Grounded Ideation
# -------------------------------------------------------------
with tab_generate:
    st.subheader("💡 Grounded Algorithmic Ideation Engine")
    st.markdown("Generate mathematically motivated, testable algorithmic RFCs grounded in codebase topology, recent baseline metrics, and negative knowledge base.")

    with st.form("ideation_form"):
        f_author = st.selectbox("Author Model:", [ModelName.GEMINI.value, ModelName.CLAUDE.value, ModelName.CHATGPT.value])
        f_strategy = st.selectbox("Target Strategy:", [s.value for s in StrategyTarget])
        f_theme = st.selectbox("Research Directive / Theme:", [
            "Volatility scaling & normalization (Parkinson / Garman-Klass)",
            "Dynamic regime filtering (Bull/Bear x Hi/Lo vol)",
            "Stop-loss & profit-target barrier calibration",
            "Quantile ranking feature expansion & volume signals",
            "Overfitting & lookahead audit fix",
            "Open discovery hypothesis",
        ])
        submitted = st.form_submit_button("Generate Proposal RFC", type="primary")

    if submitted:
        with st.spinner("Assembling context packet and constructing RFC..."):
            author_m = ModelName(f_author)
            strat_m = StrategyTarget(f_strategy)
            res_dir = Path(eval_conf.get("results_dir", ROOT_DIR / "evaluation" / "outputs" / "results"))
            
            context = ContextBuilder.build(strategy=strat_m, results_dir=res_dir, rejected_dir=pm.rejected_dir)
            next_id = pm.get_next_proposal_id()
            prompt = PromptBuilder.create_ideation_prompt(
                author_model=author_m,
                strategy=strat_m,
                theme=f_theme,
                context=context,
                next_id=next_id,
            )

            branch_map = {"gemini": "dev", "claude": "claude", "chatgpt": "chatgpt"}
            new_prop = pm.create_proposal(
                title=f"{f_theme.split('(')[0].strip()} on {f_strategy}",
                author_model=author_m,
                target_strategy=strat_m,
                assigned_branch=branch_map.get(f_author, "dev"),
                hypothesis=f"Algorithmic enhancement under research theme: {f_theme}. Targets improved precision and risk-adjusted return.",
                proposed_files=["data_access/features.py", "evaluation/configs.py"],
                implementation_spec=f"1. Implement {f_theme} in features.py\n2. Configure RunConfig in evaluation/configs.py\n3. Register --ab variant.",
            )
            ProposalStateMachine.submit_to_human(new_prop)
            pm.save_proposal(new_prop)

            st.success(f"Generated proposal **{new_prop.id}**: *{new_prop.metadata.title}*")
            st.markdown(f"Status: `{new_prop.status.value}`. Check the Review Cockpit to inspect and approve!")
            st.expander("View Grounded Ideation Prompt").code(prompt)

# -------------------------------------------------------------
# TAB 4: A/B Scorecard Visualizer
# -------------------------------------------------------------
with tab_scorecard:
    st.subheader("📊 Empirical A/B Evaluation Scorecard")
    eval_props = [p for p in all_proposals if p.scorecard_summary is not None]

    if not eval_props:
        st.info("No proposals have completed backtests yet. Run an approved proposal to generate an A/B scorecard!")
    else:
        score_options = {f"{p.id} - {p.metadata.title}": p.id for p in eval_props}
        sel_eval_key = st.selectbox("Select Backtested Proposal:", list(score_options.keys()))
        s_prop = pm.load_proposal(score_options[sel_eval_key])
        sc = s_prop.scorecard_summary

        st.markdown(f"### {s_prop.id}: {s_prop.metadata.title}")
        st.caption(f"Strategy: `{sc.get('strategy', '')}` | Variant: `{sc.get('variant', '')}` | Folds: {sc.get('num_folds', 0)}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Delta Top-1 Precision", f"{sc.get('delta_prec1_mean', 0.0):+.2f}%")
        m2.metric("Delta Total PnL", f"{sc.get('delta_pnl_total', 0.0):+.2f}%")
        m3.metric("Delta Win Rate", f"{sc.get('delta_win_rate', 0.0):+.2f}%")
        m4.metric("Fold Wins", f"{sc.get('fold_wins', 0)} / {sc.get('num_folds', 0)}")

        # Regime breakdown
        regime_deltas = sc.get("regime_deltas", {})
        if regime_deltas:
            st.markdown("#### Performance by Market Regime")
            reg_cols = st.columns(len(regime_deltas))
            for i, (reg, delta) in enumerate(regime_deltas.items()):
                reg_cols[i].metric(f"{reg}", f"{delta:+.2f}%")

        st.divider()
        if s_prop.status == ProposalStatus.EVALUATED:
            if st.button("🏆 Promote Winning Proposal to Main Branch", type="primary"):
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
                        git_mgr.sync_all_workspaces(golden_ws=golden_path, skip_key=s_prop.metadata.assigned_branch)
                        st.success(f"Proposal {s_prop.id} PROMOTED to main ({new_sha[:8]}) and all branches rebased!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Promotion failed: {e}")

# -------------------------------------------------------------
# TAB 5: Live Terminal Stream & In-Process Runner
# -------------------------------------------------------------
with tab_terminal:
    st.subheader("🖥️ In-Process Agent Execution & Live Terminal")
    st.caption("Executes the agent directly inside the harness process and streams stdout/stderr live.")

    ready_props = [p for p in all_proposals if p.status == ProposalStatus.APPROVED]
    if not ready_props:
        st.info("No proposals currently in APPROVED status. Approve a proposal in the Review Cockpit to run it here!")
    else:
        exec_options = {f"{p.id} - {p.metadata.title} (Branch: {p.metadata.assigned_branch})": p.id for p in ready_props}
        sel_exec_key = st.selectbox("Select Approved Proposal to Execute:", list(exec_options.keys()))
        target_exec_prop = pm.load_proposal(exec_options[sel_exec_key])

        if st.button("🚀 Start In-Process Agent Execution & Backtest", type="primary"):
            log_container = st.empty()
            lines = []

            def on_stream(line: str):
                lines.append(line)
                log_container.code("\n".join(lines[-40:]), language="bash")

            with st.spinner(f"Agent executing in workspace/{target_exec_prop.metadata.assigned_branch}..."):
                try:
                    res, sc = dispatcher.run_proposal_execution(
                        proposal=target_exec_prop,
                        on_output=on_stream,
                        auto_eval=True,
                    )
                    if res.success:
                        st.success(f"Execution Succeeded! Git Commit: `{res.commit_sha}`")
                        if sc:
                            st.success(f"Evaluation finished! Delta PnL: {sc.delta_total_pnl:+.2f}%")
                    else:
                        st.error(f"Execution failed: {res.error}")
                except Exception as e:
                    st.error(f"Error during execution: {e}")
