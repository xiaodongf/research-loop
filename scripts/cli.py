"""Research Loop CLI: terminal interface for managing proposals, reviews, executions, branch sync, and the 5-agent Bus."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import yaml

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.agents.dispatcher import AgentDispatcher
from src.bus.mcp_server import BusService
from src.bus.schema import EXPLICIT_TARGET_INTENTS, AgentId, Intent
from src.core.git_manager import GitManager
from src.core.ideation import ContextBuilder, PromptBuilder
from src.core.proposal_manager import ProposalManager
from src.core.schema import ModelName, ProposalStatus, ReviewVerdict, StrategyTarget
from src.core.state_machine import ProposalStateMachine
from src.harness.eval_bridge import EvalBridge


def load_config(config_path: str | None = None) -> dict:
    conf_path = Path(config_path).resolve() if config_path else (ROOT_DIR / "config.yaml")
    if conf_path.exists():
        with open(conf_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_bus_service(args: argparse.Namespace) -> BusService:
    cfg_path = Path(getattr(args, "config", None)).resolve() if getattr(args, "config", None) else None
    db_path = Path(getattr(args, "db", None)).resolve() if getattr(args, "db", None) else None
    return BusService.from_config_file(config_path=cfg_path, db_path=db_path)


def _split_agents(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


# ----------------------------------------------------------------------
# Legacy / Proposal Commands (Preserved & Upgraded for 5-Agent Config)
# ----------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> None:
    pm = ProposalManager(root_dir=ROOT_DIR)
    status_filter = ProposalStatus(args.status) if args.status else None
    proposals = pm.list_proposals(status=status_filter)

    if not proposals:
        print(f"No proposals found (filter: {args.status or 'all'}).")
        return

    print(f"\n{'ID':<10} {'STATUS':<15} {'AUTHOR':<10} {'BRANCH':<10} {'TITLE'}")
    print("-" * 75)
    for p in proposals:
        print(
            f"{p.id:<10} {p.status.value:<15} {p.author_model.value:<10} "
            f"{p.metadata.assigned_branch:<10} {p.metadata.title[:35]}"
        )
    print(f"\nTotal: {len(proposals)} proposals\n")


def cmd_show(args: argparse.Namespace) -> None:
    pm = ProposalManager(root_dir=ROOT_DIR)
    proposal = pm.load_proposal(args.id)
    if not proposal:
        print(f"Error: Proposal {args.id} not found.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Proposal: {proposal.id} - {proposal.metadata.title}")
    print(
        f"Status: {proposal.status.value} | Author: {proposal.author_model.value} | "
        f"Target: {proposal.metadata.target_strategy.value}"
    )
    print(f"{'='*70}")
    print(f"\n[Hypothesis]\n{proposal.hypothesis}")
    if proposal.proposed_files:
        print("\n[Proposed Files]\n" + "\n".join(f"  - {f}" for f in proposal.proposed_files))
    if proposal.discussion_log:
        print(f"\n[Discussion Log]\n{proposal.discussion_log}")
    if proposal.scorecard_summary:
        print("\n[Scorecard Summary]\n" + yaml.dump(proposal.scorecard_summary, default_flow_style=False))
    print(f"{'='*70}\n")


def cmd_generate(args: argparse.Namespace) -> None:
    pm = ProposalManager(root_dir=ROOT_DIR)
    config = load_config()

    author = ModelName(args.author)
    strategy = StrategyTarget(args.strategy)
    theme = args.theme

    res_dir = Path(config.get("evaluation", {}).get("results_dir", ROOT_DIR / "evaluation" / "outputs" / "results"))
    context = ContextBuilder.build(strategy=strategy, results_dir=res_dir, rejected_dir=pm.rejected_dir)
    next_id = pm.get_next_proposal_id()

    PromptBuilder.create_ideation_prompt(
        author_model=author,
        strategy=strategy,
        theme=theme,
        context=context,
        next_id=next_id,
    )

    branch_map = {
        "gemini": "dev",
        "dev": "dev",
        "claude": "claude",
        "chatgpt": "chatgpt",
        "main": "main",
        "staging": "staging",
    }
    proposal = pm.create_proposal(
        title=f"Exploration: {theme.title()} on {strategy.value}",
        author_model=author,
        target_strategy=strategy,
        assigned_branch=branch_map.get(author.value, "dev"),
        hypothesis=f"Generated research hypothesis addressing {theme}.",
        proposed_files=["data_access/features.py", "evaluation/configs.py"],
    )
    ProposalStateMachine.submit_to_human(proposal)
    pm.save_proposal(proposal)

    print(f"\n[+] Created proposal {proposal.id}: '{proposal.metadata.title}'")
    print(f"[+] Status transitioned to: {proposal.status.value}")
    print(f"[+] Review file: {pm.active_dir / f'{proposal.id}.md'}\n")


def cmd_review(args: argparse.Namespace) -> None:
    pm = ProposalManager(root_dir=ROOT_DIR)
    proposal = pm.load_proposal(args.id)
    if not proposal:
        print(f"Error: Proposal {args.id} not found.")
        sys.exit(1)

    if args.action == "approve":
        ProposalStateMachine.approve(proposal, assigned_branch=args.branch)
        pm.save_proposal(proposal)
        print(f"[+] Approved {proposal.id} for implementation on branch '{proposal.metadata.assigned_branch}'.")
    elif args.action == "reject":
        reason = args.reason or "Rejected by reviewer via CLI."
        pm.reject_proposal(proposal, reason=reason)
        print(f"[-] Rejected {proposal.id}. Moved to proposals/rejected/.")
    elif args.action == "route":
        if not args.reviewer:
            print("Error: --reviewer is required for route action (explicit routing required).")
            sys.exit(1)
        reviewer = ModelName(args.reviewer)
        ProposalStateMachine.route_to_peer(proposal, reviewer=reviewer)
        pm.save_proposal(proposal)
        print(f"[+] Routed {proposal.id} to {reviewer.value} for peer review.")


def cmd_run(args: argparse.Namespace) -> None:
    pm = ProposalManager(root_dir=ROOT_DIR)
    proposal = pm.load_proposal(args.id)
    if not proposal:
        print(f"Error: Proposal {args.id} not found.")
        sys.exit(1)

    if proposal.status != ProposalStatus.APPROVED:
        print(f"Error: Proposal must be in APPROVED status to run (current: {proposal.status.value}).")
        sys.exit(1)

    config = load_config()
    eval_conf = config.get("evaluation", {})
    eval_bridge = EvalBridge(
        python_bin=Path(eval_conf.get("python_bin", sys.executable)),
        results_dir=Path(eval_conf.get("results_dir", ROOT_DIR / "evaluation" / "outputs" / "results")),
    )
    dispatcher = AgentDispatcher(config=config, proposal_manager=pm, eval_bridge=eval_bridge)

    print(f"\n[*] Starting in-process execution for {proposal.id}...")
    exec_result, scorecard = dispatcher.run_proposal_execution(
        proposal=proposal,
        on_output=lambda line: print(f"  > {line}"),
        auto_eval=not args.no_eval,
    )

    if exec_result.success:
        print(f"\n[+] Agent execution succeeded! Commit: {exec_result.commit_sha}")
        if scorecard:
            print("[+] Evaluation Complete!")
            print(f"    - Delta Top-1 Precision: {scorecard.delta_top1_precision:+.2f}%")
            print(f"    - Delta Total PnL:       {scorecard.delta_total_pnl:+.2f}%")
            print(f"    - Fold Wins:             {scorecard.fold_wins}/{scorecard.num_folds}")
    else:
        print(f"\n[!] Execution failed: {exec_result.error}")


def cmd_sync(args: argparse.Namespace) -> None:
    config = load_config()
    ws_conf = config.get("workspaces", {})
    workspaces = {k: Path(v["path"]) for k, v in ws_conf.items() if isinstance(v, dict) and "path" in v}
    golden_key = config.get("golden_truth", {}).get("workspace", "main")
    golden_entry = ws_conf.get(golden_key) or ws_conf.get("main") or ws_conf.get("dev") or ws_conf.get("gemini") or {}
    golden_path = Path(golden_entry.get("path", ""))
    git_mgr = GitManager(workspaces=workspaces, golden_branch=config.get("golden_truth", {}).get("branch", "main"))

    print("\n[*] Synchronizing all model workspaces from golden main...")
    results = git_mgr.sync_all_workspaces(golden_ws=golden_path)
    for k, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  - {k:<10}: {status}")
    print("[+] Synchronization complete.\n")


# ----------------------------------------------------------------------
# Multi-Agent Bus Commands (`cli.py bus ...`)
# ----------------------------------------------------------------------
def cmd_bus_status(args: argparse.Namespace) -> None:
    svc = get_bus_service(args)
    fleet = svc.git_bridge.get_fleet_telemetry(store=svc.store)
    cap = svc.capacity.get_capacity_telemetry()

    if args.json:
        print(
            json.dumps(
                {
                    "fleet": [f.model_dump() for f in fleet],
                    "capacity": cap,
                },
                indent=2,
            )
        )
        return

    print("\n=== 5-Agent Fleet Telemetry ===")
    print(f"{'AGENT':<9} {'BRANCH':<10} {'HEAD':<9} {'CLEAN':<8} {'DRIFT':<14} {'UNREAD':<7} {'CLAIMS'}")
    print("-" * 82)
    for st_item in fleet:
        if args.agent and st_item.agent != AgentId.normalize(args.agent):
            continue
        short_sha = st_item.head_sha[:7] if st_item.head_sha else "missing"
        clean_lbl = "clean" if st_item.is_clean else f"dirty({st_item.dirty_count})"
        drift_lbl = f"-{st_item.behind_main}/+{st_item.ahead_main}" if st_item.exists else "n/a"
        claims_lbl = str(len(st_item.active_claims))
        print(
            f"{st_item.agent.value:<9} {st_item.branch:<10} {short_sha:<9} "
            f"{clean_lbl:<8} {drift_lbl:<14} {st_item.unread_count:<7} {claims_lbl}"
        )
        for adv in st_item.advisories:
            print(f"  ! {adv}")

    sup = cap.get("supply", {})
    active_leases = cap.get("active_leases", [])
    print("\n=== Host Compute & Evaluation Leases ===")
    print(
        f"CPU: {sup.get('ncpu')} cores (load1={sup.get('load1')}, load5={sup.get('load5')}) | "
        f"Avail RAM: {sup.get('avail_mem_gib')} GiB | Active Leases: {len(active_leases)}"
    )
    for ls in active_leases:
        prog = ls.get("fold_progress")
        prog_str = f" | folds={prog}" if prog else ""
        print(
            f"  * [{ls['lease_id']}] {ls['agent']} PID={ls['pid']} "
            f"{ls['strategy']}/{ls['mode']} ({ls['workers']}w){prog_str}"
        )
    print()


def cmd_bus_inbox(args: argparse.Namespace) -> None:
    svc = get_bus_service(args)
    agent = AgentId.normalize(args.agent)
    msgs = svc.store.get_inbox(
        agent,
        unread_only=not args.all,
        mark_read=bool(args.mark_read),
        limit=args.limit,
    )
    if args.json:
        print(json.dumps([m.model_dump() for m in msgs], indent=2))
        return
    if not msgs:
        print(f"Inbox for '{agent.value}' is empty (unread_only={not args.all}).")
        return
    print(f"\n=== Inbox for `{agent.value}` ({len(msgs)} message(s)) ===")
    for m in msgs:
        v_str = f" [{m.verdict.value}]" if m.verdict else ""
        print(f"- {m.msg_id} | {m.thread_id} R{m.round_num} | {m.from_agent.value} -> {m.intent.value}{v_str}: {m.subject}")
    print()


def cmd_bus_send(args: argparse.Namespace) -> None:
    svc = get_bus_service(args)
    to_list = _split_agents(args.to)
    cc_list = _split_agents(args.cc)
    intent_enum = Intent(args.intent)
    if intent_enum in EXPLICIT_TARGET_INTENTS and not to_list:
        print(
            f"Error: Intent '{intent_enum.value}' requires explicit non-empty --to <agents> "
            "(no default reviewer routing is allowed).",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        res = svc.send_verified(
            from_agent=args.sender,
            to_agents=to_list,
            cc_agents=cc_list,
            intent=intent_enum,
            subject=args.subject,
            body=args.body or "",
            head_sha=args.head_sha,
            base_sha=args.base_sha,
            proposal_id=args.proposal_id,
            auto_audit=not args.no_auto_audit,
            run_auto_audit_now=bool(args.run_audit_now),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        env = res["envelope"]
        print(f"[+] Sent {env['msg_id']} on thread {env['thread_id']} ({env['intent']}) -> {env['to_agents']}")
        for adv in env.get("advisories", []):
            print(f"  ! {adv}")
        for ar in res.get("auto_audit_results", []):
            print(f"  * Auto-audit [{ar['reviewer']}]: success={ar['success']} verdict={ar['verdict']} err={ar['error']}")


def cmd_bus_reply(args: argparse.Namespace) -> None:
    svc = get_bus_service(args)
    to_list = _split_agents(args.to)
    cc_list = _split_agents(args.cc)
    intent_enum = Intent(args.intent)
    if intent_enum in EXPLICIT_TARGET_INTENTS and not to_list:
        print(
            f"Error: Intent '{intent_enum.value}' requires explicit non-empty --to <agents>.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        res = svc.send_verified(
            from_agent=args.sender,
            thread_id=args.thread,
            in_reply_to=args.in_reply_to,
            to_agents=to_list,
            cc_agents=cc_list,
            intent=intent_enum,
            subject=args.subject,
            body=args.body or "",
            head_sha=args.head_sha,
            base_sha=args.base_sha,
            verdict=args.verdict,
            run_auto_audit_now=bool(args.run_audit_now),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        env = res["envelope"]
        print(f"[+] Replied {env['msg_id']} on {env['thread_id']} (R{env['round_num']}) -> {env['to_agents']}")


def cmd_bus_thread(args: argparse.Namespace) -> None:
    svc = get_bus_service(args)
    bundle = svc.store.get_thread(args.thread_id)
    if not bundle:
        print(f"Error: Thread '{args.thread_id}' not found.", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(
            json.dumps(
                {
                    "thread": bundle["thread"].model_dump(),
                    "findings": [f.model_dump() for f in bundle["findings"]],
                    "messages": [m.model_dump() for m in bundle["messages"]],
                    "mirror_path": bundle["mirror_path"],
                },
                indent=2,
            )
        )
        return
    mirror = Path(bundle["mirror_path"])
    if mirror.exists():
        print(mirror.read_text(encoding="utf-8"))
    else:
        print(f"Thread {args.thread_id}: {len(bundle['messages'])} messages")


def cmd_bus_audit_run(args: argparse.Namespace) -> None:
    from src.bus.audit_worker import write_audit_job_meta
    from src.bus.schema import utc_now_iso

    svc = get_bus_service(args)
    runs_dir = svc.store.db_path.parent / "audit_runs"

    def _emit_line(ln: str) -> None:
        print(f"  [audit] {ln}", flush=True)

    if args.msg_id:
        records = svc.audit_worker.execute_audit_for_message(
            args.msg_id,
            on_output=_emit_line,
        )
    else:
        records = svc.audit_worker.poll_and_execute_once(
            limit=args.limit,
            on_output=_emit_line,
        )
    out = [
        {
            "reviewer": r.reviewer.value,
            "success": r.success,
            "reply_msg_id": r.reply_envelope.msg_id if r.reply_envelope else None,
            "verdict": r.reply_envelope.verdict.value if (r.reply_envelope and r.reply_envelope.verdict) else None,
            "duration_seconds": round(r.duration_seconds, 2),
            "error": r.error,
        }
        for r in records
    ]
    if args.msg_id:
        meta_file = runs_dir / f"{args.msg_id}.json"
        existing_meta = {}
        if meta_file.exists():
            try:
                existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                existing_meta = {}
        all_ok = bool(out) and all(item["success"] for item in out)
        existing_meta.update(
            {
                "msg_id": args.msg_id,
                "status": "COMPLETED" if all_ok else "FAILED",
                "finished_at": utc_now_iso(),
                "results": out,
            }
        )
        write_audit_job_meta(runs_dir, args.msg_id, existing_meta)

    if args.json:
        print(json.dumps(out, indent=2), flush=True)
    else:
        if not out:
            print("No pending read-only auto-audit messages to execute.", flush=True)
        for item in out:
            print(
                f"[audit] reviewer={item['reviewer']} success={item['success']} "
                f"verdict={item['verdict']} reply={item['reply_msg_id']} err={item['error']}",
                flush=True,
            )


def cmd_bus_sync_knowledge(args: argparse.Namespace) -> None:
    svc = get_bus_service(args)
    stats = svc.knowledge.sync_all_workspaces()
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(
            f"[+] Knowledge Sync Complete: scanned={stats['scanned']} "
            f"added={stats['added']} deduped={stats['deduped']}"
        )


def cmd_bus_search(args: argparse.Namespace) -> None:
    svc = get_bus_service(args)
    hits = svc.store.search_knowledge(
        args.query,
        category=args.category,
        source_agent=args.agent,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps([h.model_dump() for h in hits], indent=2))
        return
    if not hits:
        print(f"No knowledge entries matched query '{args.query}'.")
        return
    print(f"\n=== Knowledge Search: '{args.query}' ({len(hits)} hit(s)) ===")
    for h in hits:
        print(f"- [{h.entry_id}] ({h.category} | {h.source_agent.value}) {h.title}")
        print(f"  {h.summary[:180]}")
        if h.artifact_path:
            print(f"  path: {h.artifact_path}")
    print()


def cmd_bus_watch(args: argparse.Namespace) -> None:
    svc = get_bus_service(args)
    print(f"[*] Watching Bus for pending read-only auto-audits (interval={args.interval}s)...")
    while True:
        records = svc.audit_worker.poll_and_execute_once(
            limit=args.limit,
            on_output=lambda ln: print(f"  [audit] {ln}"),
        )
        for r in records:
            v = r.reply_envelope.verdict.value if (r.reply_envelope and r.reply_envelope.verdict) else "ERR"
            print(f"[+] Auto-audit completed: reviewer={r.reviewer.value} success={r.success} verdict={v}")
        if args.once:
            break
        time.sleep(max(0.5, float(args.interval)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research Loop: Multi-Model Trading Research Harness & Agent Bus")
    parser.add_argument("--config", help="Optional override path to config.yaml")
    parser.add_argument("--db", help="Optional override path to SQLite hub.db")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List proposals")
    p_list.add_argument("--status", choices=[s.value for s in ProposalStatus], help="Filter by status")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = subparsers.add_parser("show", help="Show proposal details")
    p_show.add_argument("id", help="Proposal ID (e.g. PROP-001)")
    p_show.set_defaults(func=cmd_show)

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate new proposal")
    p_gen.add_argument("--author", choices=[m.value for m in ModelName], default="gemini")
    p_gen.add_argument("--strategy", choices=[s.value for s in StrategyTarget], default="standalone_tb")
    p_gen.add_argument("--theme", default="Volatility scaling & regime filter")
    p_gen.set_defaults(func=cmd_generate)

    # review
    p_rev = subparsers.add_parser("review", help="Review proposal")
    p_rev.add_argument("id", help="Proposal ID")
    p_rev.add_argument("action", choices=["approve", "reject", "route"])
    p_rev.add_argument("--branch", help="Assigned branch for approve action")
    p_rev.add_argument("--reviewer", choices=[m.value for m in ModelName], help="Reviewer for route action")
    p_rev.add_argument("--reason", help="Rejection rationale")
    p_rev.set_defaults(func=cmd_review)

    # run
    p_run = subparsers.add_parser("run", help="Execute approved proposal")
    p_run.add_argument("id", help="Proposal ID")
    p_run.add_argument("--no-eval", action="store_true", help="Skip evaluation backtest")
    p_run.set_defaults(func=cmd_run)

    # sync
    p_sync = subparsers.add_parser("sync", help="Synchronize model workspaces from main")
    p_sync.set_defaults(func=cmd_sync)

    # bus
    p_bus = subparsers.add_parser("bus", help="5-agent Research Bus CLI")
    p_bus.add_argument("--config", help="Optional override path to config.yaml")
    p_bus.add_argument("--db", help="Optional override path to SQLite hub.db")
    bus_sub = p_bus.add_subparsers(dest="bus_cmd", required=True)

    b_status = bus_sub.add_parser("status", help="Show 5-checkout fleet status, drift, and host capacity")
    b_status.add_argument("--agent", help="Filter to a single agent")
    b_status.add_argument("--json", action="store_true", help="Emit JSON")
    b_status.set_defaults(func=cmd_bus_status)

    b_inbox = bus_sub.add_parser("inbox", help="Inspect agent inbox")
    b_inbox.add_argument("--agent", required=True, help="Target agent (main|staging|dev|claude|chatgpt|human)")
    b_inbox.add_argument("--all", action="store_true", help="Include already-read messages")
    b_inbox.add_argument("--mark-read", action="store_true", help="Mark returned messages as read")
    b_inbox.add_argument("--limit", type=int, default=50)
    b_inbox.add_argument("--json", action="store_true")
    b_inbox.set_defaults(func=cmd_bus_inbox)

    b_send = bus_sub.add_parser("send", help="Create thread / send message on the Bus")
    b_send.add_argument("--from", dest="sender", required=True, help="Sender agent ID")
    b_send.add_argument("--to", default="", help="Comma-separated recipient agent IDs (explicitly required for audits)")
    b_send.add_argument("--cc", default="", help="Comma-separated CC agent IDs")
    b_send.add_argument("--intent", required=True, choices=[i.value for i in Intent])
    b_send.add_argument("--subject", required=True)
    b_send.add_argument("--body", default="")
    b_send.add_argument("--head-sha", default=None)
    b_send.add_argument("--base-sha", default=None)
    b_send.add_argument("--proposal-id", default=None)
    b_send.add_argument("--no-auto-audit", action="store_true", help="Deliver to session inbox only without auto-audit")
    b_send.add_argument("--run-audit-now", action="store_true", help="Execute headless read-only audit synchronously now")
    b_send.add_argument("--json", action="store_true")
    b_send.set_defaults(func=cmd_bus_send)

    b_reply = bus_sub.add_parser("reply", help="Reply to an existing Bus thread")
    b_reply.add_argument("--from", dest="sender", required=True)
    b_reply.add_argument("--thread", required=True, help="Thread ID (e.g. THR-0001)")
    b_reply.add_argument("--in-reply-to", default=None)
    b_reply.add_argument("--to", default="", help="Comma-separated recipient agent IDs")
    b_reply.add_argument("--cc", default="", help="Comma-separated CC agent IDs")
    b_reply.add_argument("--intent", required=True, choices=[i.value for i in Intent])
    b_reply.add_argument("--subject", required=True)
    b_reply.add_argument("--body", default="")
    b_reply.add_argument("--verdict", default=None)
    b_reply.add_argument("--head-sha", default=None)
    b_reply.add_argument("--base-sha", default=None)
    b_reply.add_argument("--run-audit-now", action="store_true")
    b_reply.add_argument("--json", action="store_true")
    b_reply.set_defaults(func=cmd_bus_reply)

    b_thr = bus_sub.add_parser("thread", help="Read structured thread timeline & Markdown mirror")
    b_thr.add_argument("thread_id", help="Thread ID (e.g. THR-0001)")
    b_thr.add_argument("--json", action="store_true")
    b_thr.set_defaults(func=cmd_bus_thread)

    b_aud = bus_sub.add_parser("audit-run", help="Execute pending read-only headless audits in /tmp worktrees")
    b_aud.add_argument("--msg-id", default=None, help="Specific message ID to audit")
    b_aud.add_argument("--limit", type=int, default=5)
    b_aud.add_argument("--json", action="store_true")
    b_aud.set_defaults(func=cmd_bus_audit_run)

    b_sync_k = bus_sub.add_parser("sync-knowledge", help="Index research_notes/ and reports/ across all 5 checkouts")
    b_sync_k.add_argument("--json", action="store_true")
    b_sync_k.set_defaults(func=cmd_bus_sync_knowledge)

    b_search = bus_sub.add_parser("search", help="Search cross-checkout FTS5 knowledge base")
    b_search.add_argument("query", help="Search query terms")
    b_search.add_argument("--category", default=None)
    b_search.add_argument("--agent", default=None)
    b_search.add_argument("--limit", type=int, default=20)
    b_search.add_argument("--json", action="store_true")
    b_search.set_defaults(func=cmd_bus_search)

    b_watch = bus_sub.add_parser("watch", help="Watch bus and execute pending read-only auto-audits")
    b_watch.add_argument("--interval", type=float, default=5.0)
    b_watch.add_argument("--limit", type=int, default=5)
    b_watch.add_argument("--once", action="store_true", help="Poll once and exit")
    b_watch.set_defaults(func=cmd_bus_watch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
