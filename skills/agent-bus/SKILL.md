---
name: agent-bus
description: Coordinate code audits, peer RFC critiques, staging handoffs, host evaluation leases, and shared research knowledge across the 5 trading checkouts (main, staging, dev, claude, chatgpt) without copy-pasting or mutating peer workspaces.
---

# Multi-Agent Research Bus (`research_loop`)

Use the `research-loop-bus` MCP tools (or `/home/xiaodong/workspace/research_loop/venv/bin/python /home/xiaodong/workspace/research_loop/scripts/cli.py bus ...`) whenever coordinating across `main` (`trading-exp`), `staging` (`trading-staging`), `dev` (`trading`), `claude` (`trading-claude`), and `chatgpt` (`trading-gpt`).

## Golden Rules

1. **Always Check `bus_status` First**: At session start or before heavy work, call `bus_status` to check unread messages, advisory `behind_main` drift, active peer file claims, and host CPU/RAM leases.
2. **Explicit `to_agents` Required**: Never rely on default reviewer routing. Every `AUDIT_REQUEST`, `PROPOSAL_RFC`, `REMEDIATION_READY`, or `INTEGRATION_REQUEST` **must** specify non-empty `to_agents` (e.g., `["claude", "chatgpt"]` or `["staging"]`).
3. **Immutable Commit Handoffs Only**: Commit all code changes in `head_sha` before sending `AUDIT_REQUEST`, `REMEDIATION_READY`, or `INTEGRATION_REQUEST`. Uncommitted working-tree edits on changed files are rejected by `verify_commit_handoff`.
4. **Zero Checkout Pollution**: Never `cd` into or edit another agent's checkout directory. Use `bus_inspect_commit` (with `worktree=true` if full file context is needed) to inspect peer commits via `/tmp/rl-worktrees/`.
5. **Coordinate Heavy A/B Runs**: Acquire an atomic lease via `bus_acquire_eval_lease` before launching `evaluation.cli` / walk-forward backtests, and release it (`release_lease_id`) when finished.
6. **Search & Preserve Institutional Memory**: Before starting a new feature/hypothesis, call `bus_knowledge` (`action="search"`) to check prior `research_notes/` and negative A/B results across all 5 branches. After any concluded A/B (especially failed/unmerged ones), publish the result with `bus_knowledge` (`action="publish"`).

## MCP Tool Quick Reference

- `bus_status(mark_read=false)` — Unread inbox, `behind_main` advisory, peer `work_claims`, host CPU/RAM leases.
- `bus_claim_work(files=[...], summary="...")` — Advisory file-overlap registration (informs peers, never blocks).
- `bus_send(to_agents=[...], intent="AUDIT_REQUEST", subject="...", body="...", head_sha="...", base_sha="...")` — Start a verified thread.
- `bus_reply(thread_id="THR-0001", to_agents=[...], intent="AUDIT_VERDICT", verdict="CHANGES_REQUESTED", findings=[...])` — Reply with structured `P0`/`P1`/`P2` findings or `REMEDIATION_READY` (`resolved_findings={"THR-0001-F1": "<new_sha>"}`).
- `bus_read_thread(thread_id="THR-0001")` — Full message timeline, `P0`/`P1`/`P2` checklist, and Markdown mirror path.
- `bus_inspect_commit(msg_id="MSG-000001", worktree=false)` — Read commit diff via `refs/peers/*` or mount a clean `/tmp` worktree.
- `bus_acquire_eval_lease(strategy="standalone_tb", mode="quick", workers=4)` — Atomic host capacity guard + waitlist.
- `bus_knowledge(action="search"|"publish"|"sync", query="...")` — Cross-checkout FTS5 institutional memory.
