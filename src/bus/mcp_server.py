"""Identity-bound Stdio JSON-RPC 2.0 MCP Server & unified BusService facade."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bus.audit_worker import AutoAuditWorker
from src.bus.capacity_lease import CapacityLeaseManager
from src.bus.git_bridge import GitBridge
from src.bus.knowledge_sync import KnowledgeManager
from src.bus.schema import (
    AgentId,
    AuditVerdictType,
    CodeContext,
    FindingInput,
    Intent,
)
from src.bus.store import BusStore


class IdentitySpoofingError(PermissionError):
    """Raised when an MCP client bound to one checkout attempts to send messages as another agent."""


class BusService:
    """High-level facade wiring BusStore, GitBridge, CapacityLeaseManager, KnowledgeManager, and AutoAuditWorker."""

    def __init__(
        self,
        *,
        root_dir: Path,
        config: Dict[str, Any],
        db_path: Optional[Path] = None,
        peer_refs_dir: Optional[Path] = None,
        worktree_root: Optional[Path] = None,
    ):
        self.root_dir = root_dir.resolve()
        self.config = config
        bus_conf = config.get("bus", {})

        eff_db = Path(db_path) if db_path else (self.root_dir / bus_conf.get("db_path", "state/hub.db"))
        eff_threads = self.root_dir / bus_conf.get("threads_dir", "threads")
        eff_know = self.root_dir / bus_conf.get("knowledge_dir", "knowledge")
        eff_peer_refs = (
            Path(peer_refs_dir)
            if peer_refs_dir
            else (self.root_dir / bus_conf.get("peer_refs_dir", "state/peer_refs.git"))
        )
        eff_wt_root = Path(worktree_root) if worktree_root else Path(bus_conf.get("worktree_root", "/tmp/rl-worktrees"))

        self.store = BusStore(
            db_path=eff_db,
            threads_dir=eff_threads,
            knowledge_dir=eff_know,
        )
        golden_conf = config.get("golden_truth", {})
        self.git_bridge = GitBridge(
            workspaces=config.get("workspaces", {}),
            peer_refs_dir=eff_peer_refs,
            worktree_root=eff_wt_root,
            golden_agent=AgentId.normalize(golden_conf.get("workspace", "main")),
            golden_branch=golden_conf.get("branch", "main"),
        )
        eval_conf = config.get("evaluation", {})
        self.capacity = CapacityLeaseManager(
            self.store,
            capacity_script_path=eval_conf.get("capacity_script"),
        )
        self.knowledge = KnowledgeManager(
            self.store,
            workspaces=config.get("workspaces", {}),
        )
        self.audit_worker = AutoAuditWorker(
            self.store,
            self.git_bridge,
            config=config,
            timeout_seconds=float(bus_conf.get("audit_timeout_seconds", 300.0)),
        )

    @classmethod
    def from_config_file(
        cls,
        config_path: Optional[Path] = None,
        *,
        db_path: Optional[Path] = None,
    ) -> "BusService":
        root_dir = Path(__file__).resolve().parent.parent.parent
        c_path = Path(config_path).resolve() if config_path else (root_dir / "config.yaml")
        conf = yaml.safe_load(c_path.read_text(encoding="utf-8")) if c_path.exists() else {}
        return cls(root_dir=c_path.parent if config_path else root_dir, config=conf, db_path=db_path)

    def send_verified(
        self,
        *,
        from_agent: AgentId | str,
        to_agents: Sequence[AgentId | str],
        intent: Intent | str,
        subject: str,
        body: str = "",
        thread_id: Optional[str] = None,
        in_reply_to: Optional[str] = None,
        cc_agents: Optional[Sequence[AgentId | str]] = None,
        head_sha: Optional[str] = None,
        base_sha: Optional[str] = None,
        workspace_path: Optional[Path | str] = None,
        verdict: Optional[AuditVerdictType | str] = None,
        new_findings: Optional[Sequence[FindingInput | Dict[str, Any]]] = None,
        resolved_findings: Optional[Dict[str, str]] = None,
        verified_findings: Optional[Sequence[str]] = None,
        wontfix_findings: Optional[Sequence[str]] = None,
        proposal_id: Optional[str] = None,
        auto_audit: bool = True,
        run_auto_audit_now: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sender = AgentId.normalize(from_agent)
        code_ctx: Optional[CodeContext] = None
        advisories: List[str] = []

        _, _, drift_adv = self.git_bridge.get_agent_drift_advisory(sender, store=self.store)
        if drift_adv:
            advisories.append(drift_adv)

        if head_sha:
            code_ctx = self.git_bridge.verify_commit_handoff(
                agent=sender,
                head_sha=head_sha,
                base_sha=base_sha,
                workspace_path=workspace_path,
            )

        envelope = self.store.send_message(
            from_agent=sender,
            to_agents=to_agents,
            cc_agents=cc_agents,
            intent=intent,
            subject=subject,
            body=body,
            thread_id=thread_id,
            in_reply_to=in_reply_to,
            code_context=code_ctx,
            verdict=verdict,
            new_findings=new_findings,
            resolved_findings=resolved_findings,
            verified_findings=verified_findings,
            wontfix_findings=wontfix_findings,
            proposal_id=proposal_id,
            auto_audit=auto_audit,
            metadata=metadata,
            extra_advisories=advisories,
        )

        audit_records = []
        if run_auto_audit_now:
            audit_records = self.audit_worker.execute_audit_for_message(envelope)

        return {
            "envelope": envelope.model_dump(),
            "auto_audit_results": [
                {
                    "reviewer": r.reviewer.value,
                    "success": r.success,
                    "reply_msg_id": r.reply_envelope.msg_id if r.reply_envelope else None,
                    "verdict": r.reply_envelope.verdict.value if (r.reply_envelope and r.reply_envelope.verdict) else None,
                    "error": r.error,
                }
                for r in audit_records
            ],
        }


MCP_TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "bus_status",
        "description": "Returns unread inbox messages for this checkout, advisory branch drift vs main, active peer file claims, and host CPU/RAM evaluation lease status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mark_read": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "bus_send",
        "description": "Creates a new cross-agent thread or sends a typed message (AUDIT_REQUEST, PROPOSAL_RFC, INTEGRATION_REQUEST, BASELINE_UPDATED, ADVISORY_NOTE). Explicit non-empty to_agents is mandatory for review/audit/integration intents.",
        "inputSchema": {
            "type": "object",
            "required": ["to_agents", "intent", "subject"],
            "properties": {
                "to_agents": {"type": "array", "items": {"type": "string"}},
                "cc_agents": {"type": "array", "items": {"type": "string"}},
                "intent": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string", "default": ""},
                "head_sha": {"type": "string"},
                "base_sha": {"type": "string"},
                "proposal_id": {"type": "string"},
                "auto_audit": {"type": "boolean", "default": True},
                "run_auto_audit_now": {"type": "boolean", "default": False},
                "from_agent": {"type": "string"},
            },
        },
    },
    {
        "name": "bus_reply",
        "description": "Replies to an existing thread (AUDIT_VERDICT, PEER_CRITIQUE, REMEDIATION_READY, STAGING_RESULT) and updates P0/P1/P2 finding statuses.",
        "inputSchema": {
            "type": "object",
            "required": ["thread_id", "to_agents", "intent", "subject"],
            "properties": {
                "thread_id": {"type": "string"},
                "in_reply_to": {"type": "string"},
                "to_agents": {"type": "array", "items": {"type": "string"}},
                "cc_agents": {"type": "array", "items": {"type": "string"}},
                "intent": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string", "default": ""},
                "head_sha": {"type": "string"},
                "base_sha": {"type": "string"},
                "verdict": {"type": "string"},
                "findings": {"type": "array", "items": {"type": "object"}},
                "resolved_findings": {"type": "object"},
                "verified_findings": {"type": "array", "items": {"type": "string"}},
                "wontfix_findings": {"type": "array", "items": {"type": "string"}},
                "run_auto_audit_now": {"type": "boolean", "default": False},
                "from_agent": {"type": "string"},
            },
        },
    },
    {
        "name": "bus_read_thread",
        "description": "Reads full structured thread timeline, P0/P1/P2 findings checklist, and Markdown mirror path.",
        "inputSchema": {
            "type": "object",
            "required": ["thread_id"],
            "properties": {
                "thread_id": {"type": "string"},
                "mark_read": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "bus_inspect_commit",
        "description": "Inspects a peer commit via refs/peers/* without modifying any user checkout, and optionally creates a clean disposable /tmp worktree.",
        "inputSchema": {
            "type": "object",
            "required": ["msg_id"],
            "properties": {
                "msg_id": {"type": "string"},
                "worktree": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "bus_claim_work",
        "description": "Registers an advisory active work claim on files and returns non-blocking overlap/drift advisories.",
        "inputSchema": {
            "type": "object",
            "required": ["files"],
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string", "default": ""},
                "thread_id": {"type": "string"},
                "release": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "bus_acquire_eval_lease",
        "description": "Atomically acquires or releases a host CPU/RAM evaluation lease wrapping check_host_capacity.py.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "release_lease_id": {"type": "string"},
                "strategy": {"type": "string", "default": "standalone_tb"},
                "mode": {"type": "string", "default": "quick"},
                "workers": {"type": "number", "default": 4},
                "thread_id": {"type": "string"},
                "checkpoint_dir": {"type": "string"},
                "ignore_host_guard": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "bus_knowledge",
        "description": "Searches or publishes to the shared cross-checkout FTS5 Knowledge Store (research_notes, reports, positive & negative A/Bs).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search", "publish", "sync"], "default": "search"},
                "query": {"type": "string", "default": ""},
                "category": {"type": "string"},
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "outcome": {"type": "string", "default": "NEGATIVE"},
                "strategy": {"type": "string", "default": "standalone_tb"},
                "commit_sha": {"type": "string"},
                "failure_conditions": {"type": "string", "default": ""},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
]


class BusMCPServer:
    """Stdio JSON-RPC 2.0 MCP Server bound to a single checkout identity (`--agent`)."""

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(
        self,
        service: BusService,
        *,
        bound_agent: AgentId,
        bound_workspace: Optional[Path] = None,
    ):
        self.service = service
        self.bound_agent = bound_agent
        self.bound_workspace = (
            Path(bound_workspace).resolve()
            if bound_workspace
            else self.service.git_bridge.get_workspace_path(bound_agent)
        )

    def _enforce_identity(self, args: Dict[str, Any]) -> AgentId:
        requested = args.get("from_agent")
        if requested is not None and str(requested).strip() != "":
            norm = AgentId.normalize(requested)
            if norm != self.bound_agent:
                raise IdentitySpoofingError(
                    f"Identity violation: MCP server is bound to '{self.bound_agent.value}' "
                    f"and cannot publish as '{norm.value}'."
                )
        return self.bound_agent

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = dict(arguments or {})

        if name == "bus_status":
            inbox = self.service.store.get_inbox(
                self.bound_agent,
                unread_only=True,
                mark_read=bool(args.get("mark_read", False)),
            )
            behind, ahead, drift_adv = self.service.git_bridge.get_agent_drift_advisory(
                self.bound_agent, store=self.service.store
            )
            peer_claims = [
                c.model_dump()
                for c in self.service.store.list_active_claims()
                if c.agent != self.bound_agent
            ]
            cap = self.service.capacity.get_capacity_telemetry()
            return {
                "agent": self.bound_agent.value,
                "workspace": str(self.bound_workspace) if self.bound_workspace else None,
                "unread_count": len(inbox),
                "unread_messages": [m.model_dump() for m in inbox],
                "behind_main": behind,
                "ahead_main": ahead,
                "drift_advisory": drift_adv,
                "peer_active_claims": peer_claims,
                "host_capacity": cap,
            }

        if name == "bus_send":
            sender = self._enforce_identity(args)
            return self.service.send_verified(
                from_agent=sender,
                to_agents=args.get("to_agents") or [],
                cc_agents=args.get("cc_agents") or [],
                intent=args["intent"],
                subject=args["subject"],
                body=args.get("body", ""),
                head_sha=args.get("head_sha"),
                base_sha=args.get("base_sha"),
                workspace_path=self.bound_workspace,
                proposal_id=args.get("proposal_id"),
                auto_audit=bool(args.get("auto_audit", True)),
                run_auto_audit_now=bool(args.get("run_auto_audit_now", False)),
            )

        if name == "bus_reply":
            sender = self._enforce_identity(args)
            return self.service.send_verified(
                from_agent=sender,
                thread_id=args["thread_id"],
                in_reply_to=args.get("in_reply_to"),
                to_agents=args.get("to_agents") or [],
                cc_agents=args.get("cc_agents") or [],
                intent=args["intent"],
                subject=args["subject"],
                body=args.get("body", ""),
                head_sha=args.get("head_sha"),
                base_sha=args.get("base_sha"),
                workspace_path=self.bound_workspace,
                verdict=args.get("verdict"),
                new_findings=args.get("findings"),
                resolved_findings=args.get("resolved_findings"),
                verified_findings=args.get("verified_findings"),
                wontfix_findings=args.get("wontfix_findings"),
                auto_audit=bool(args.get("intent") == Intent.REMEDIATION_READY.value),
                run_auto_audit_now=bool(args.get("run_auto_audit_now", False)),
            )

        if name == "bus_read_thread":
            tid = args["thread_id"]
            bundle = self.service.store.get_thread(tid)
            if not bundle:
                raise KeyError(f"Thread '{tid}' not found.")
            if args.get("mark_read", True):
                mids = [m.msg_id for m in bundle["messages"]]
                self.service.store.mark_messages_read(self.bound_agent, mids)
            return {
                "thread": bundle["thread"].model_dump(),
                "findings": [f.model_dump() for f in bundle["findings"]],
                "messages": [m.model_dump() for m in bundle["messages"]],
                "mirror_path": bundle["mirror_path"],
            }

        if name == "bus_inspect_commit":
            msg = self.service.store.get_message(args["msg_id"])
            if not msg or not msg.code_context:
                raise ValueError(f"Message '{args.get('msg_id')}' has no attached code_context.")
            info = self.service.git_bridge.inspect_commit(
                from_agent=msg.from_agent,
                thread_id=msg.thread_id,
                head_sha=msg.code_context.head_sha,
                base_sha=msg.code_context.base_sha,
                source_workspace=msg.code_context.source_workspace,
            )
            if args.get("worktree"):
                wt = self.service.git_bridge.create_disposable_worktree(
                    head_sha=msg.code_context.head_sha,
                    thread_id=msg.thread_id,
                    tag=self.bound_agent.value,
                    source_workspace=msg.code_context.source_workspace,
                )
                info["worktree_path"] = str(wt)
            return info

        if name == "bus_claim_work":
            if args.get("release"):
                released = self.service.store.release_work_claims(self.bound_agent)
                return {"released": released}
            claim, adv = self.service.store.claim_work(
                agent=self.bound_agent,
                files=args.get("files") or [],
                summary=args.get("summary", ""),
                thread_id=args.get("thread_id"),
            )
            _, _, drift_adv = self.service.git_bridge.get_agent_drift_advisory(
                self.bound_agent, store=self.service.store
            )
            if drift_adv:
                adv.append(drift_adv)
            return {"claim": claim.model_dump(), "advisories": adv}

        if name == "bus_acquire_eval_lease":
            if args.get("release_lease_id"):
                ok = self.service.capacity.release_eval_lease(str(args["release_lease_id"]))
                return {"released": ok, "lease_id": args["release_lease_id"]}
            _, _, drift_adv = self.service.git_bridge.get_agent_drift_advisory(
                self.bound_agent, store=self.service.store
            )
            advs = [drift_adv] if drift_adv else []
            res = self.service.capacity.acquire_eval_lease(
                agent=self.bound_agent,
                workspace_path=str(self.bound_workspace or ""),
                strategy=args.get("strategy", "standalone_tb"),
                mode=args.get("mode", "quick"),
                workers=float(args.get("workers", 4.0)),
                thread_id=args.get("thread_id"),
                checkpoint_dir=args.get("checkpoint_dir"),
                ignore_host_guard=bool(args.get("ignore_host_guard", False)),
                advisories=advs,
            )
            return res.model_dump()

        if name in ("bus_knowledge", "bus_search_knowledge", "bus_publish_knowledge"):
            action = args.get("action", "publish" if name == "bus_publish_knowledge" else "search")
            if action == "sync":
                return self.service.knowledge.sync_all_workspaces()
            if action == "publish":
                entry = self.service.knowledge.publish_experiment_record(
                    source_agent=self.bound_agent,
                    title=args["title"],
                    summary=args["summary"],
                    outcome=args.get("outcome", "NEGATIVE"),
                    strategy=args.get("strategy", "standalone_tb"),
                    commit_sha=args.get("commit_sha"),
                    failure_conditions=args.get("failure_conditions", ""),
                    tags=args.get("tags"),
                )
                return entry.model_dump()
            hits = self.service.store.search_knowledge(
                query=args.get("query", ""),
                category=args.get("category"),
            )
            return {"count": len(hits), "results": [h.model_dump() for h in hits]}

        raise KeyError(f"Unknown MCP tool: {name}")

    def handle_jsonrpc_message(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = req.get("method", "")
        req_id = req.get("id")
        params = req.get("params") or {}

        # JSON-RPC notifications (no id) require no response
        if req_id is None and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                client_ver = params.get("protocolVersion", self.PROTOCOL_VERSION)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": client_ver,
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": f"research-loop-bus-{self.bound_agent.value}",
                            "version": "1.0.0",
                        },
                    },
                }
            if method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS_SPEC}}
            if method == "tools/call":
                t_name = params.get("name", "")
                t_args = params.get("arguments") or {}
                try:
                    payload = self.call_tool(t_name, t_args)
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                            "isError": False,
                        },
                    }
                except Exception as tool_exc:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {"error": type(tool_exc).__name__, "detail": str(tool_exc)}
                                    ),
                                }
                            ],
                            "isError": True,
                        },
                    }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(exc)},
            }

    def serve_stdio(self) -> None:
        """Runs newline-delimited JSON-RPC 2.0 loop over stdin/stdout."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line or line.lower().startswith("content-length:"):
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle_jsonrpc_message(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description="Research Loop Identity-Bound MCP Bus Server")
    ap.add_argument("--agent", required=True, help="Bound agent identity (main, staging, dev, claude, chatgpt)")
    ap.add_argument("--workspace", default=None, help="Bound workspace checkout directory")
    ap.add_argument("--config", default=None, help="Path to research_loop/config.yaml")
    ap.add_argument("--db", default=None, help="Override path to SQLite hub.db")
    args = ap.parse_args()

    svc = BusService.from_config_file(
        config_path=Path(args.config) if args.config else None,
        db_path=Path(args.db) if args.db else None,
    )
    server = BusMCPServer(
        svc,
        bound_agent=AgentId.normalize(args.agent),
        bound_workspace=Path(args.workspace) if args.workspace else None,
    )
    server.serve_stdio()


if __name__ == "__main__":
    main()
