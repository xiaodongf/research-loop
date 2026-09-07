"""Research Loop CLI: terminal interface for managing proposals, reviews, executions, and branch sync."""
from __future__ import annotations
import sys, os, argparse, yaml
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.schema import ModelName, StrategyTarget, ProposalStatus, ReviewVerdict
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


def cmd_list(args):
    pm = ProposalManager(root_dir=ROOT_DIR)
    status_filter = ProposalStatus(args.status) if args.status else None
    proposals = pm.list_proposals(status=status_filter)

    if not proposals:
        print(f"No proposals found (filter: {args.status or 'all'}).")
        return

    print(f"\n{'ID':<10} {'STATUS':<15} {'AUTHOR':<10} {'BRANCH':<10} {'TITLE'}")
    print("-" * 75)
    for p in proposals:
        print(f"{p.id:<10} {p.status.value:<15} {p.author_model.value:<10} {p.metadata.assigned_branch:<10} {p.metadata.title[:35]}")
    print(f"\nTotal: {len(proposals)} proposals\n")


def cmd_show(args):
    pm = ProposalManager(root_dir=ROOT_DIR)
    proposal = pm.load_proposal(args.id)
    if not proposal:
        print(f"Error: Proposal {args.id} not found.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Proposal: {proposal.id} - {proposal.metadata.title}")
    print(f"Status: {proposal.status.value} | Author: {proposal.author_model.value} | Target: {proposal.metadata.target_strategy.value}")
    print(f"{'='*70}")
    print(f"\n[Hypothesis]\n{proposal.hypothesis}")
    if proposal.proposed_files:
        print(f"\n[Proposed Files]\n" + "\n".join(f"  - {f}" for f in proposal.proposed_files))
    if proposal.discussion_log:
        print(f"\n[Discussion Log]\n{proposal.discussion_log}")
    if proposal.scorecard_summary:
        print(f"\n[Scorecard Summary]\n" + yaml.dump(proposal.scorecard_summary, default_flow_style=False))
    print(f"{'='*70}\n")


def cmd_generate(args):
    pm = ProposalManager(root_dir=ROOT_DIR)
    config = load_config()

    author = ModelName(args.author)
    strategy = StrategyTarget(args.strategy)
    theme = args.theme

    # Build context
    res_dir = Path(config.get("evaluation", {}).get("results_dir", ROOT_DIR / "evaluation" / "outputs" / "results"))
    context = ContextBuilder.build(strategy=strategy, results_dir=res_dir, rejected_dir=pm.rejected_dir)
    next_id = pm.get_next_proposal_id()

    prompt = PromptBuilder.create_ideation_prompt(
        author_model=author,
        strategy=strategy,
        theme=theme,
        context=context,
        next_id=next_id,
    )

    branch_map = {"gemini": "dev", "claude": "claude", "chatgpt": "chatgpt"}
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


def cmd_review(args):
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
            print("Error: --reviewer (gemini|claude|chatgpt) is required for route action.")
            sys.exit(1)
        reviewer = ModelName(args.reviewer)
        ProposalStateMachine.route_to_peer(proposal, reviewer=reviewer)
        pm.save_proposal(proposal)
        print(f"[+] Routed {proposal.id} to {reviewer.value} for peer review.")


def cmd_run(args):
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
            print(f"[+] Evaluation Complete!")
            print(f"    - Delta Top-1 Precision: {scorecard.delta_top1_precision:+.2f}%")
            print(f"    - Delta Total PnL:       {scorecard.delta_total_pnl:+.2f}%")
            print(f"    - Fold Wins:             {scorecard.fold_wins}/{scorecard.num_folds}")
    else:
        print(f"\n[!] Execution failed: {exec_result.error}")


def cmd_sync(args):
    config = load_config()
    workspaces = {k: Path(v["path"]) for k, v in config.get("workspaces", {}).items()}
    golden_path = Path(config.get("workspaces", {}).get("gemini", {}).get("path", ""))
    git_mgr = GitManager(workspaces=workspaces, golden_branch=config.get("golden_truth", {}).get("branch", "main"))

    print(f"\n[*] Synchronizing all model workspaces from golden main...")
    results = git_mgr.sync_all_workspaces(golden_ws=golden_path)
    for k, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  - {k:<10}: {status}")
    print("[+] Synchronization complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Research Loop: Multi-Model Trading Research Harness")
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
    p_gen.add_argument("--author", choices=["gemini", "claude", "chatgpt"], default="gemini")
    p_gen.add_argument("--strategy", choices=["standalone_tb", "quantile", "stacked_tb", "hybrid"], default="standalone_tb")
    p_gen.add_argument("--theme", default="Volatility scaling & regime filter")
    p_gen.set_defaults(func=cmd_generate)

    # review
    p_rev = subparsers.add_parser("review", help="Review proposal")
    p_rev.add_argument("id", help="Proposal ID")
    p_rev.add_argument("action", choices=["approve", "reject", "route"])
    p_rev.add_argument("--branch", help="Assigned branch for approve action")
    p_rev.add_argument("--reviewer", choices=["gemini", "claude", "chatgpt"], help="Reviewer for route action")
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
