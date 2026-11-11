"""SQLite WAL ACID persistence engine, FTS5 knowledge index, and atomic Markdown thread mirrors."""
from __future__ import annotations
import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from src.bus.schema import (
    EXPLICIT_TARGET_INTENTS,
    WORKSPACE_AGENTS,
    AgentId,
    AuditVerdictType,
    CodeContext,
    Finding,
    FindingInput,
    FindingSeverity,
    FindingStatus,
    Intent,
    KnowledgeEntry,
    MessageEnvelope,
    ThreadRecord,
    ThreadStatus,
    WorkClaim,
    utc_now_iso,
)


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:max_len].strip("-") or "thread"


class BusStore:
    """Zero-daemon multi-process ACID bus store backed by SQLite WAL + Markdown thread mirrors."""

    def __init__(
        self,
        db_path: Path | str,
        threads_dir: Optional[Path | str] = None,
        knowledge_dir: Optional[Path | str] = None,
    ):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        base_root = self.db_path.parent.parent
        self.threads_dir = Path(threads_dir).resolve() if threads_dir else (base_root / "threads").resolve()
        self.knowledge_dir = Path(knowledge_dir).resolve() if knowledge_dir else (base_root / "knowledge").resolve()
        self.threads_dir.mkdir(parents=True, exist_ok=True)
        (self.knowledge_dir / "experiments").mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,  # Explicit transaction control for BEGIN IMMEDIATE
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def immediate_tx(self, max_retries: int = 25) -> Iterator[sqlite3.Connection]:
        """Acquires a multi-process-safe SQLite write lock using BEGIN IMMEDIATE with jittered retry."""
        conn = self._connect()
        try:
            for attempt in range(max_retries):
                try:
                    conn.execute("BEGIN IMMEDIATE;")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == max_retries - 1:
                        raise
                    time.sleep(0.01 * (attempt + 1) + (os.getpid() % 7) * 0.002)
            try:
                yield conn
                if conn.in_transaction:
                    conn.execute("COMMIT;")
            except Exception:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK;")
                except sqlite3.Error:
                    pass
                raise
        finally:
            conn.close()

    @contextmanager
    def read_conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS counters (
                name TEXT PRIMARY KEY,
                val INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                created_by TEXT NOT NULL,
                proposal_id TEXT,
                latest_round INTEGER NOT NULL DEFAULT 1,
                max_auto_rounds INTEGER NOT NULL DEFAULT 3,
                auto_audit INTEGER NOT NULL DEFAULT 1,
                mirror_filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS messages (
                msg_id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                round_num INTEGER NOT NULL,
                in_reply_to TEXT,
                from_agent TEXT NOT NULL,
                to_agents_json TEXT NOT NULL,
                cc_agents_json TEXT NOT NULL,
                intent TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                code_context_json TEXT,
                verdict TEXT,
                resolved_findings_json TEXT NOT NULL DEFAULT '{}',
                verified_findings_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                advisories_json TEXT NOT NULL DEFAULT '[]',
                auto_dispatched INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_messages_thread_seq ON messages(thread_id, seq)",
            """
            CREATE TABLE IF NOT EXISTS inbox_receipts (
                msg_id TEXT NOT NULL REFERENCES messages(msg_id),
                agent TEXT NOT NULL,
                is_cc INTEGER NOT NULL DEFAULT 0,
                read_at TEXT,
                PRIMARY KEY (msg_id, agent)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_inbox_agent_unread ON inbox_receipts(agent, read_at)",
            """
            CREATE TABLE IF NOT EXISTS findings (
                finding_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL REFERENCES threads(thread_id),
                round_num INTEGER NOT NULL,
                author_agent TEXT NOT NULL,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                file TEXT NOT NULL,
                lines TEXT NOT NULL,
                problem TEXT NOT NULL,
                suggested_fix TEXT NOT NULL,
                status TEXT NOT NULL,
                resolved_in_sha TEXT,
                verified_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_findings_thread ON findings(thread_id)",
            """
            CREATE TABLE IF NOT EXISTS work_claims (
                claim_id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                thread_id TEXT,
                files_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS baseline_state (
                branch TEXT PRIMARY KEY,
                head_sha TEXT NOT NULL,
                prev_sha TEXT,
                summary TEXT NOT NULL DEFAULT '',
                touches_ssot_or_models INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS eval_leases (
                lease_id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                pid INTEGER NOT NULL,
                workspace_path TEXT NOT NULL DEFAULT '',
                strategy TEXT NOT NULL,
                mode TEXT NOT NULL,
                workers REAL NOT NULL,
                threads_per_worker REAL NOT NULL,
                mem_per_worker_gib REAL NOT NULL,
                thread_id TEXT,
                checkpoint_dir TEXT,
                acquired_at TEXT NOT NULL,
                acquired_epoch REAL NOT NULL,
                released_at TEXT,
                status TEXT NOT NULL DEFAULT 'ACTIVE'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS eval_waitlist (
                wait_id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                pid INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                mode TEXT NOT NULL,
                workers REAL NOT NULL,
                thread_id TEXT,
                requested_at TEXT NOT NULL,
                notified_at TEXT,
                status TEXT NOT NULL DEFAULT 'WAITING'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                entry_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                source_branch TEXT NOT NULL DEFAULT '',
                commit_sha TEXT,
                strategy TEXT,
                outcome TEXT,
                summary TEXT NOT NULL,
                body_excerpt TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                artifact_path TEXT NOT NULL DEFAULT '',
                content_sha256 TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                entry_id UNINDEXED,
                title,
                category,
                strategy,
                tags,
                summary,
                body_excerpt
            )
            """,
        ]
        with self.immediate_tx() as conn:
            for stmt in ddl_statements:
                conn.execute(stmt)

    @staticmethod
    def _next_counter(conn: sqlite3.Connection, name: str) -> int:
        row = conn.execute("SELECT val FROM counters WHERE name = ?", (name,)).fetchone()
        nxt = (int(row["val"]) + 1) if row else 1
        conn.execute(
            "INSERT INTO counters(name, val) VALUES(?, ?) "
            "ON CONFLICT(name) DO UPDATE SET val = excluded.val",
            (name, nxt),
        )
        return nxt

    # ------------------------------------------------------------------
    # Advisory Checks (Decision #3: Never block, always inform)
    # ------------------------------------------------------------------
    def compute_file_overlap_advisories(
        self,
        caller_agent: AgentId,
        candidate_files: Sequence[str],
        exclude_thread_id: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[str]:
        norm_files = {f.strip().lstrip("./") for f in candidate_files if f and f.strip()}
        if not norm_files:
            return []

        def _query(c: sqlite3.Connection) -> List[str]:
            rows = c.execute(
                "SELECT agent, thread_id, files_json, summary FROM work_claims WHERE active = 1 AND agent != ?",
                (caller_agent.value,),
            ).fetchall()
            warnings: List[str] = []
            for r in rows:
                if exclude_thread_id and r["thread_id"] == exclude_thread_id:
                    continue
                claimed = {x.strip().lstrip("./") for x in json.loads(r["files_json"])}
                overlap = sorted(norm_files & claimed)
                if overlap:
                    thr_part = f" (thread {r['thread_id']})" if r["thread_id"] else ""
                    warnings.append(
                        f"Advisory: peer '{r['agent']}'{thr_part} has active work claim overlapping "
                        f"{overlap}: {r['summary'] or 'in-progress edits'}."
                    )
            return warnings

        if conn is not None:
            return _query(conn)
        with self.read_conn() as c:
            return _query(c)

    # ------------------------------------------------------------------
    # Send & Reply (Thread + Envelope + Findings Lifecycle)
    # ------------------------------------------------------------------
    def send_message(
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
        code_context: Optional[CodeContext | Dict[str, Any]] = None,
        verdict: Optional[AuditVerdictType | str] = None,
        new_findings: Optional[Sequence[FindingInput | Dict[str, Any]]] = None,
        resolved_findings: Optional[Dict[str, str]] = None,
        verified_findings: Optional[Sequence[str]] = None,
        wontfix_findings: Optional[Sequence[str]] = None,
        proposal_id: Optional[str] = None,
        auto_audit: bool = True,
        max_auto_rounds: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        extra_advisories: Optional[Sequence[str]] = None,
    ) -> MessageEnvelope:
        sender = AgentId.normalize(from_agent)
        intent_enum = Intent(intent) if not isinstance(intent, Intent) else intent
        to_list = [AgentId.normalize(a) for a in (to_agents or []) if a is not None and str(a).strip() != ""]
        cc_list = [
            AgentId.normalize(a)
            for a in (cc_agents or [])
            if a is not None and str(a).strip() != "" and AgentId.normalize(a) not in to_list
        ]

        # Pre-validate Decision #5 before opening DB write lock
        if intent_enum in EXPLICIT_TARGET_INTENTS and not to_list:
            raise ValueError(
                f"Intent '{intent_enum.value}' requires explicit non-empty 'to_agents' "
                f"(no default reviewer routing is allowed)."
            )

        code_ctx_obj: Optional[CodeContext] = None
        if isinstance(code_context, CodeContext):
            code_ctx_obj = code_context
        elif isinstance(code_context, dict):
            code_ctx_obj = CodeContext(**code_context)

        verdict_enum: Optional[AuditVerdictType] = None
        if verdict is not None:
            verdict_enum = AuditVerdictType(verdict) if not isinstance(verdict, AuditVerdictType) else verdict

        now = utc_now_iso()

        with self.immediate_tx() as conn:
            if thread_id is None:
                t_num = self._next_counter(conn, "thread")
                thread_id = f"THR-{t_num:04d}"
                mirror_filename = f"{thread_id}-{_slugify(subject)}.md"
                round_num = 1
                initial_status = self._derive_thread_status(
                    current_status=ThreadStatus.OPEN,
                    intent=intent_enum,
                    verdict=verdict_enum,
                    open_blocking_count=0,
                )
                conn.execute(
                    """
                    INSERT INTO threads(
                        thread_id, title, status, created_by, proposal_id,
                        latest_round, max_auto_rounds, auto_audit, mirror_filename,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        subject,
                        initial_status.value,
                        sender.value,
                        proposal_id,
                        round_num,
                        max_auto_rounds,
                        1 if auto_audit else 0,
                        mirror_filename,
                        now,
                        now,
                    ),
                )
            else:
                t_row = conn.execute(
                    "SELECT * FROM threads WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()
                if not t_row:
                    raise KeyError(f"Thread '{thread_id}' does not exist.")
                round_num = int(t_row["latest_round"])
                if intent_enum == Intent.REMEDIATION_READY:
                    round_num += 1

            seq = self._next_counter(conn, "message")
            msg_id = f"MSG-{seq:06d}"

            # Compute advisory overlap warnings (non-blocking per Decision #3)
            advisories = list(extra_advisories or [])
            if code_ctx_obj and code_ctx_obj.changed_files:
                advisories.extend(
                    self.compute_file_overlap_advisories(
                        caller_agent=sender,
                        candidate_files=code_ctx_obj.changed_files,
                        exclude_thread_id=thread_id,
                        conn=conn,
                    )
                )

            # Update existing findings first (resolutions / verifications / wontfix)
            resolved_map = dict(resolved_findings or {})
            for fid, sha in resolved_map.items():
                conn.execute(
                    """
                    UPDATE findings
                    SET status = ?, resolved_in_sha = ?, updated_at = ?
                    WHERE finding_id = ? AND thread_id = ?
                    """,
                    (FindingStatus.FIXED_PENDING_VERIFY.value, sha, now, fid, thread_id),
                )

            verified_ids = list(verified_findings or [])
            for fid in verified_ids:
                conn.execute(
                    """
                    UPDATE findings
                    SET status = ?, verified_by = ?, updated_at = ?
                    WHERE finding_id = ? AND thread_id = ?
                    """,
                    (FindingStatus.VERIFIED.value, sender.value, now, fid, thread_id),
                )

            for fid in (wontfix_findings or []):
                conn.execute(
                    """
                    UPDATE findings
                    SET status = ?, updated_at = ?
                    WHERE finding_id = ? AND thread_id = ?
                    """,
                    (FindingStatus.WONTFIX.value, now, fid, thread_id),
                )

            # Create newly reported findings on this thread
            created_findings: List[Finding] = []
            for raw_f in (new_findings or []):
                f_in = raw_f if isinstance(raw_f, FindingInput) else FindingInput(**raw_f)
                f_idx = self._next_counter(conn, f"finding:{thread_id}")
                fid = f"{thread_id}-F{f_idx}"
                f_obj = Finding(
                    id=fid,
                    thread_id=thread_id,
                    round_num=round_num,
                    author_agent=sender,
                    severity=f_in.severity,
                    category=f_in.category,
                    file=f_in.file,
                    lines=f_in.lines,
                    problem=f_in.problem,
                    suggested_fix=f_in.suggested_fix,
                    status=FindingStatus.OPEN,
                    created_at=now,
                    updated_at=now,
                )
                conn.execute(
                    """
                    INSERT INTO findings(
                        finding_id, thread_id, round_num, author_agent, severity,
                        category, file, lines, problem, suggested_fix, status,
                        resolved_in_sha, verified_by, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f_obj.id,
                        f_obj.thread_id,
                        f_obj.round_num,
                        f_obj.author_agent.value,
                        f_obj.severity.value,
                        f_obj.category,
                        f_obj.file,
                        f_obj.lines,
                        f_obj.problem,
                        f_obj.suggested_fix,
                        f_obj.status.value,
                        None,
                        None,
                        f_obj.created_at,
                        f_obj.updated_at,
                    ),
                )
                created_findings.append(f_obj)

            envelope = MessageEnvelope(
                msg_id=msg_id,
                thread_id=thread_id,
                round_num=round_num,
                in_reply_to=in_reply_to,
                from_agent=sender,
                to_agents=to_list,
                cc_agents=cc_list,
                intent=intent_enum,
                subject=subject,
                body=body,
                code_context=code_ctx_obj,
                verdict=verdict_enum,
                findings=created_findings,
                resolved_findings=resolved_map,
                verified_findings=verified_ids,
                metadata=dict(metadata or {}),
                advisories=advisories,
                created_at=now,
            )

            conn.execute(
                """
                INSERT INTO messages(
                    msg_id, seq, thread_id, round_num, in_reply_to, from_agent,
                    to_agents_json, cc_agents_json, intent, subject, body,
                    code_context_json, verdict, resolved_findings_json,
                    verified_findings_json, metadata_json, advisories_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.msg_id,
                    seq,
                    envelope.thread_id,
                    envelope.round_num,
                    envelope.in_reply_to,
                    envelope.from_agent.value,
                    json.dumps([a.value for a in envelope.to_agents]),
                    json.dumps([a.value for a in envelope.cc_agents]),
                    envelope.intent.value,
                    envelope.subject,
                    envelope.body,
                    envelope.code_context.model_dump_json() if envelope.code_context else None,
                    envelope.verdict.value if envelope.verdict else None,
                    json.dumps(envelope.resolved_findings),
                    json.dumps(envelope.verified_findings),
                    json.dumps(envelope.metadata),
                    json.dumps(envelope.advisories),
                    envelope.created_at,
                ),
            )

            # Deliver per-recipient unread inbox receipts
            recipients_to = [a for a in envelope.to_agents if a != sender]
            recipients_cc = [a for a in envelope.cc_agents if a != sender and a not in recipients_to]
            if intent_enum == Intent.BASELINE_UPDATED and not recipients_to:
                recipients_to = [a for a in WORKSPACE_AGENTS if a != sender]

            for r_agent in recipients_to:
                conn.execute(
                    "INSERT OR IGNORE INTO inbox_receipts(msg_id, agent, is_cc, read_at) VALUES (?, ?, 0, NULL)",
                    (envelope.msg_id, r_agent.value),
                )
            for r_agent in recipients_cc:
                conn.execute(
                    "INSERT OR IGNORE INTO inbox_receipts(msg_id, agent, is_cc, read_at) VALUES (?, ?, 1, NULL)",
                    (envelope.msg_id, r_agent.value),
                )

            # Record BASELINE_UPDATED if applicable
            if intent_enum == Intent.BASELINE_UPDATED and code_ctx_obj:
                conn.execute(
                    """
                    INSERT INTO baseline_state(branch, head_sha, prev_sha, summary, touches_ssot_or_models, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(branch) DO UPDATE SET
                        head_sha = excluded.head_sha,
                        prev_sha = excluded.prev_sha,
                        summary = excluded.summary,
                        touches_ssot_or_models = excluded.touches_ssot_or_models,
                        updated_at = excluded.updated_at
                    """,
                    (
                        code_ctx_obj.branch or "main",
                        code_ctx_obj.head_sha,
                        code_ctx_obj.base_sha,
                        subject,
                        1 if envelope.metadata.get("touches_ssot_or_models") else 0,
                        now,
                    ),
                )

            # Update thread status & round
            cur_t = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            open_blocking = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM findings
                WHERE thread_id = ? AND status IN ('OPEN', 'FIXED_PENDING_VERIFY')
                  AND severity IN ('P0', 'P1')
                """,
                (thread_id,),
            ).fetchone()["cnt"]
            next_status = self._derive_thread_status(
                current_status=ThreadStatus(cur_t["status"]),
                intent=intent_enum,
                verdict=verdict_enum,
                open_blocking_count=int(open_blocking),
            )
            if round_num > int(cur_t["max_auto_rounds"]) and next_status == ThreadStatus.CHANGES_REQUESTED:
                next_status = ThreadStatus.ESCALATED

            conn.execute(
                "UPDATE threads SET status = ?, latest_round = ?, updated_at = ? WHERE thread_id = ?",
                (next_status.value, round_num, now, thread_id),
            )

        # Render atomic Markdown mirror outside write lock (reading committed snapshot)
        self._render_thread_markdown(thread_id)
        return envelope

    @staticmethod
    def _derive_thread_status(
        *,
        current_status: ThreadStatus,
        intent: Intent,
        verdict: Optional[AuditVerdictType],
        open_blocking_count: int,
    ) -> ThreadStatus:
        if intent == Intent.ESCALATE_HUMAN:
            return ThreadStatus.ESCALATED
        if intent in (Intent.AUDIT_REQUEST, Intent.PROPOSAL_RFC, Intent.REMEDIATION_READY):
            return ThreadStatus.AWAITING_AUDIT
        if intent == Intent.INTEGRATION_REQUEST:
            return ThreadStatus.IN_STAGING
        if intent == Intent.STAGING_RESULT:
            if verdict in (AuditVerdictType.APPROVED, AuditVerdictType.APPROVED_WITH_CAUTION):
                return ThreadStatus.APPROVED
            if verdict in (AuditVerdictType.CHANGES_REQUESTED, AuditVerdictType.BLOCKED, AuditVerdictType.REJECTED):
                return ThreadStatus.CHANGES_REQUESTED
        if intent in (Intent.AUDIT_VERDICT, Intent.PEER_CRITIQUE):
            if verdict in (AuditVerdictType.CHANGES_REQUESTED, AuditVerdictType.NEEDS_REVISION, AuditVerdictType.BLOCKED, AuditVerdictType.REJECTED):
                return ThreadStatus.CHANGES_REQUESTED
            if verdict in (AuditVerdictType.APPROVED, AuditVerdictType.APPROVED_WITH_CAUTION) and open_blocking_count == 0:
                return ThreadStatus.APPROVED
        return current_status

    # ------------------------------------------------------------------
    # Thread, Inbox & Finding Queries
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_finding(r: sqlite3.Row) -> Finding:
        return Finding(
            id=r["finding_id"],
            thread_id=r["thread_id"],
            round_num=int(r["round_num"]),
            author_agent=AgentId.normalize(r["author_agent"]),
            severity=FindingSeverity(r["severity"]),
            category=r["category"],
            file=r["file"],
            lines=r["lines"],
            problem=r["problem"],
            suggested_fix=r["suggested_fix"],
            status=FindingStatus(r["status"]),
            resolved_in_sha=r["resolved_in_sha"],
            verified_by=AgentId.normalize(r["verified_by"]) if r["verified_by"] else None,
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    def _row_to_message(self, conn: sqlite3.Connection, r: sqlite3.Row) -> MessageEnvelope:
        code_ctx = json.loads(r["code_context_json"]) if r["code_context_json"] else None
        f_rows = conn.execute(
            "SELECT * FROM findings WHERE thread_id = ? AND round_num = ? AND author_agent = ? AND created_at = ?",
            (r["thread_id"], r["round_num"], r["from_agent"], r["created_at"]),
        ).fetchall()
        return MessageEnvelope(
            msg_id=r["msg_id"],
            thread_id=r["thread_id"],
            round_num=int(r["round_num"]),
            in_reply_to=r["in_reply_to"],
            from_agent=AgentId.normalize(r["from_agent"]),
            to_agents=[AgentId.normalize(x) for x in json.loads(r["to_agents_json"])],
            cc_agents=[AgentId.normalize(x) for x in json.loads(r["cc_agents_json"])],
            intent=Intent(r["intent"]),
            subject=r["subject"],
            body=r["body"],
            code_context=CodeContext(**code_ctx) if code_ctx else None,
            verdict=AuditVerdictType(r["verdict"]) if r["verdict"] else None,
            findings=[self._row_to_finding(fr) for fr in f_rows],
            resolved_findings=json.loads(r["resolved_findings_json"] or "{}"),
            verified_findings=json.loads(r["verified_findings_json"] or "[]"),
            metadata=json.loads(r["metadata_json"] or "{}"),
            advisories=json.loads(r["advisories_json"] or "[]"),
            created_at=r["created_at"],
        )

    @staticmethod
    def _row_to_thread(r: sqlite3.Row) -> ThreadRecord:
        return ThreadRecord(
            thread_id=r["thread_id"],
            title=r["title"],
            status=ThreadStatus(r["status"]),
            created_by=AgentId.normalize(r["created_by"]),
            proposal_id=r["proposal_id"],
            latest_round=int(r["latest_round"]),
            max_auto_rounds=int(r["max_auto_rounds"]),
            auto_audit=bool(r["auto_audit"]),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )

    def get_message(self, msg_id: str) -> Optional[MessageEnvelope]:
        with self.read_conn() as conn:
            r = conn.execute("SELECT * FROM messages WHERE msg_id = ?", (msg_id,)).fetchone()
            return self._row_to_message(conn, r) if r else None

    def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        with self.read_conn() as conn:
            t_row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            if not t_row:
                return None
            m_rows = conn.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY seq ASC",
                (thread_id,),
            ).fetchall()
            f_rows = conn.execute(
                "SELECT * FROM findings WHERE thread_id = ? ORDER BY finding_id ASC",
                (thread_id,),
            ).fetchall()
            mirror_path = self.threads_dir / t_row["mirror_filename"]
            return {
                "thread": self._row_to_thread(t_row),
                "messages": [self._row_to_message(conn, mr) for mr in m_rows],
                "findings": [self._row_to_finding(fr) for fr in f_rows],
                "mirror_path": str(mirror_path),
            }

    def list_threads(self, status: Optional[ThreadStatus] = None, limit: int = 100) -> List[ThreadRecord]:
        with self.read_conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM threads WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM threads ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_thread(r) for r in rows]

    def update_thread_status(self, thread_id: str, status: ThreadStatus) -> None:
        now = utc_now_iso()
        with self.immediate_tx() as conn:
            conn.execute(
                "UPDATE threads SET status = ?, updated_at = ? WHERE thread_id = ?",
                (status.value, now, thread_id),
            )
        self._render_thread_markdown(thread_id)

    def update_finding_status(
        self,
        finding_id: str,
        status: FindingStatus,
        *,
        resolved_in_sha: Optional[str] = None,
        verified_by: Optional[AgentId | str] = None,
    ) -> Optional[Finding]:
        now = utc_now_iso()
        ver_str = AgentId.normalize(verified_by).value if verified_by else None
        thread_id: Optional[str] = None
        with self.immediate_tx() as conn:
            row = conn.execute("SELECT thread_id FROM findings WHERE finding_id = ?", (finding_id,)).fetchone()
            if not row:
                return None
            thread_id = row["thread_id"]
            conn.execute(
                """
                UPDATE findings
                SET status = ?,
                    resolved_in_sha = COALESCE(?, resolved_in_sha),
                    verified_by = COALESCE(?, verified_by),
                    updated_at = ?
                WHERE finding_id = ?
                """,
                (status.value, resolved_in_sha, ver_str, now, finding_id),
            )
            updated = conn.execute("SELECT * FROM findings WHERE finding_id = ?", (finding_id,)).fetchone()
            res = self._row_to_finding(updated)
        if thread_id:
            self._render_thread_markdown(thread_id)
        return res

    def get_inbox(
        self,
        agent: AgentId | str,
        *,
        unread_only: bool = True,
        mark_read: bool = False,
        limit: int = 50,
    ) -> List[MessageEnvelope]:
        aid = AgentId.normalize(agent)
        if mark_read:
            with self.immediate_tx() as conn:
                clause = "AND ir.read_at IS NULL" if unread_only else ""
                rows = conn.execute(
                    f"""
                    SELECT m.* FROM inbox_receipts ir
                    JOIN messages m ON m.msg_id = ir.msg_id
                    WHERE ir.agent = ? {clause}
                    ORDER BY m.seq ASC LIMIT ?
                    """,
                    (aid.value, limit),
                ).fetchall()
                envelopes = [self._row_to_message(conn, r) for r in rows]
                if envelopes:
                    now = utc_now_iso()
                    conn.executemany(
                        "UPDATE inbox_receipts SET read_at = ? WHERE msg_id = ? AND agent = ? AND read_at IS NULL",
                        [(now, e.msg_id, aid.value) for e in envelopes],
                    )
                return envelopes

        with self.read_conn() as conn:
            clause = "AND ir.read_at IS NULL" if unread_only else ""
            rows = conn.execute(
                f"""
                SELECT m.* FROM inbox_receipts ir
                JOIN messages m ON m.msg_id = ir.msg_id
                WHERE ir.agent = ? {clause}
                ORDER BY m.seq ASC LIMIT ?
                """,
                (aid.value, limit),
            ).fetchall()
            return [self._row_to_message(conn, r) for r in rows]

    def mark_messages_read(self, agent: AgentId | str, msg_ids: Sequence[str]) -> int:
        aid = AgentId.normalize(agent)
        if not msg_ids:
            return 0
        now = utc_now_iso()
        with self.immediate_tx() as conn:
            cur = conn.executemany(
                "UPDATE inbox_receipts SET read_at = ? WHERE msg_id = ? AND agent = ? AND read_at IS NULL",
                [(now, mid, aid.value) for mid in msg_ids],
            )
            return cur.rowcount if cur.rowcount is not None else 0

    def get_unread_counts(self) -> Dict[str, int]:
        counts = {a.value: 0 for a in WORKSPACE_AGENTS}
        counts[AgentId.HUMAN.value] = 0
        with self.read_conn() as conn:
            rows = conn.execute(
                "SELECT agent, COUNT(*) AS cnt FROM inbox_receipts WHERE read_at IS NULL GROUP BY agent"
            ).fetchall()
            for r in rows:
                counts[r["agent"]] = int(r["cnt"])
        return counts

    def claim_pending_auto_audits(self, limit: int = 10) -> List[MessageEnvelope]:
        """Atomically claims un-dispatched read-only review messages whose thread enables auto_audit."""
        with self.immediate_tx() as conn:
            rows = conn.execute(
                """
                SELECT m.* FROM messages m
                JOIN threads t ON t.thread_id = m.thread_id
                WHERE m.auto_dispatched = 0
                  AND t.auto_audit = 1
                  AND m.intent IN ('AUDIT_REQUEST', 'PROPOSAL_RFC', 'REMEDIATION_READY')
                  AND m.round_num <= t.max_auto_rounds
                ORDER BY m.seq ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            envelopes = [self._row_to_message(conn, r) for r in rows]
            if envelopes:
                conn.executemany(
                    "UPDATE messages SET auto_dispatched = 1 WHERE msg_id = ?",
                    [(e.msg_id,) for e in envelopes],
                )
            return envelopes

    def mark_auto_dispatched(self, msg_id: str) -> None:
        with self.immediate_tx() as conn:
            conn.execute("UPDATE messages SET auto_dispatched = 1 WHERE msg_id = ?", (msg_id,))

    # ------------------------------------------------------------------
    # Advisory Work Claims (Decision #3)
    # ------------------------------------------------------------------
    def claim_work(
        self,
        *,
        agent: AgentId | str,
        files: Sequence[str],
        summary: str = "",
        thread_id: Optional[str] = None,
        replace_previous: bool = True,
    ) -> tuple[WorkClaim, List[str]]:
        aid = AgentId.normalize(agent)
        clean_files = [f.strip().lstrip("./") for f in files if f and f.strip()]
        now = utc_now_iso()
        with self.immediate_tx() as conn:
            if replace_previous:
                conn.execute(
                    "UPDATE work_claims SET active = 0 WHERE agent = ? AND active = 1",
                    (aid.value,),
                )
            c_num = self._next_counter(conn, "claim")
            cid = f"CLM-{c_num:05d}"
            advisories = self.compute_file_overlap_advisories(
                caller_agent=aid,
                candidate_files=clean_files,
                exclude_thread_id=thread_id,
                conn=conn,
            )
            conn.execute(
                """
                INSERT INTO work_claims(claim_id, agent, thread_id, files_json, summary, created_at, active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (cid, aid.value, thread_id, json.dumps(clean_files), summary, now),
            )
            claim = WorkClaim(
                claim_id=cid,
                agent=aid,
                thread_id=thread_id,
                files=clean_files,
                summary=summary,
                created_at=now,
                active=True,
            )
            return claim, advisories

    def release_work_claims(self, agent: AgentId | str, claim_id: Optional[str] = None) -> int:
        aid = AgentId.normalize(agent)
        with self.immediate_tx() as conn:
            if claim_id:
                cur = conn.execute(
                    "UPDATE work_claims SET active = 0 WHERE claim_id = ? AND agent = ?",
                    (claim_id, aid.value),
                )
            else:
                cur = conn.execute(
                    "UPDATE work_claims SET active = 0 WHERE agent = ? AND active = 1",
                    (aid.value,),
                )
            return cur.rowcount or 0

    def list_active_claims(self, agent: Optional[AgentId | str] = None) -> List[WorkClaim]:
        with self.read_conn() as conn:
            if agent:
                aid = AgentId.normalize(agent)
                rows = conn.execute(
                    "SELECT * FROM work_claims WHERE active = 1 AND agent = ? ORDER BY created_at DESC",
                    (aid.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM work_claims WHERE active = 1 ORDER BY created_at DESC"
                ).fetchall()
            return [
                WorkClaim(
                    claim_id=r["claim_id"],
                    agent=AgentId.normalize(r["agent"]),
                    thread_id=r["thread_id"],
                    files=json.loads(r["files_json"]),
                    summary=r["summary"],
                    created_at=r["created_at"],
                    active=bool(r["active"]),
                )
                for r in rows
            ]

    def get_latest_baseline(self, branch: str = "main") -> Optional[Dict[str, Any]]:
        with self.read_conn() as conn:
            row = conn.execute("SELECT * FROM baseline_state WHERE branch = ?", (branch,)).fetchone()
            if not row:
                return None
            return dict(row)

    # ------------------------------------------------------------------
    # Hybrid Knowledge Store + FTS5 Search (Decision #4)
    # ------------------------------------------------------------------
    def upsert_knowledge(
        self,
        *,
        title: str,
        category: str,
        source_agent: AgentId | str,
        summary: str,
        body_excerpt: str = "",
        source_branch: str = "",
        commit_sha: Optional[str] = None,
        strategy: Optional[str] = None,
        outcome: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        artifact_path: str = "",
        content_sha256: Optional[str] = None,
    ) -> tuple[KnowledgeEntry, bool]:
        """Inserts or deduplicates a knowledge entry by SHA256. Returns (entry, created_new)."""
        aid = AgentId.normalize(source_agent)
        tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
        normalized_blob = f"{category}|{title.strip()}|{summary.strip()}|{body_excerpt.strip()}"
        digest = content_sha256 or hashlib.sha256(normalized_blob.encode("utf-8")).hexdigest()
        now = utc_now_iso()

        with self.immediate_tx() as conn:
            existing = conn.execute(
                "SELECT * FROM knowledge_entries WHERE content_sha256 = ?",
                (digest,),
            ).fetchone()
            if existing:
                return self._row_to_knowledge(existing), False

            k_num = self._next_counter(conn, "knowledge")
            eid = f"KNW-{k_num:05d}"
            entry = KnowledgeEntry(
                entry_id=eid,
                title=title.strip(),
                category=category.strip(),
                source_agent=aid,
                source_branch=source_branch,
                commit_sha=commit_sha,
                strategy=strategy,
                outcome=outcome,
                summary=summary.strip(),
                body_excerpt=body_excerpt[:6000],
                tags=tag_list,
                artifact_path=artifact_path,
                content_sha256=digest,
                created_at=now,
            )
            conn.execute(
                """
                INSERT INTO knowledge_entries(
                    entry_id, title, category, source_agent, source_branch,
                    commit_sha, strategy, outcome, summary, body_excerpt,
                    tags_json, artifact_path, content_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.title,
                    entry.category,
                    entry.source_agent.value,
                    entry.source_branch,
                    entry.commit_sha,
                    entry.strategy,
                    entry.outcome,
                    entry.summary,
                    entry.body_excerpt,
                    json.dumps(entry.tags),
                    entry.artifact_path,
                    entry.content_sha256,
                    entry.created_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO knowledge_fts(entry_id, title, category, strategy, tags, summary, body_excerpt)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.title,
                    entry.category,
                    entry.strategy or "",
                    " ".join(entry.tags),
                    entry.summary,
                    entry.body_excerpt,
                ),
            )
            return entry, True

    @staticmethod
    def _row_to_knowledge(r: sqlite3.Row) -> KnowledgeEntry:
        return KnowledgeEntry(
            entry_id=r["entry_id"],
            title=r["title"],
            category=r["category"],
            source_agent=AgentId.normalize(r["source_agent"]),
            source_branch=r["source_branch"],
            commit_sha=r["commit_sha"],
            strategy=r["strategy"],
            outcome=r["outcome"],
            summary=r["summary"],
            body_excerpt=r["body_excerpt"],
            tags=json.loads(r["tags_json"] or "[]"),
            artifact_path=r["artifact_path"],
            content_sha256=r["content_sha256"],
            created_at=r["created_at"],
        )

    def search_knowledge(
        self,
        query: str = "",
        *,
        category: Optional[str] = None,
        source_agent: Optional[AgentId | str] = None,
        limit: int = 30,
    ) -> List[KnowledgeEntry]:
        tokens = [re.sub(r"[^a-zA-Z0-9_]+", "", tok) for tok in query.strip().split()]
        tokens = [t for t in tokens if t]

        filters: List[str] = []
        params: List[Any] = []
        if category:
            filters.append("ke.category = ?")
            params.append(category)
        if source_agent:
            filters.append("ke.source_agent = ?")
            params.append(AgentId.normalize(source_agent).value)
        where_extra = (" AND " + " AND ".join(filters)) if filters else ""

        with self.read_conn() as conn:
            if not tokens:
                sql = f"SELECT * FROM knowledge_entries ke WHERE 1=1 {where_extra} ORDER BY ke.created_at DESC LIMIT ?"
                rows = conn.execute(sql, [*params, limit]).fetchall()
                return [self._row_to_knowledge(r) for r in rows]

            fts_expr = " OR ".join(f'"{t}"*' for t in tokens)
            sql = f"""
                SELECT ke.*
                FROM knowledge_fts fts
                JOIN knowledge_entries ke ON ke.entry_id = fts.entry_id
                WHERE knowledge_fts MATCH ? {where_extra}
                ORDER BY bm25(knowledge_fts) ASC, ke.created_at DESC
                LIMIT ?
            """
            rows = conn.execute(sql, [fts_expr, *params, limit]).fetchall()
            return [self._row_to_knowledge(r) for r in rows]

    # ------------------------------------------------------------------
    # Atomic Markdown Thread Mirror Renderer
    # ------------------------------------------------------------------
    def _render_thread_markdown(self, thread_id: str) -> Optional[Path]:
        with self.read_conn() as conn:
            t_row = conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            if not t_row:
                return None
            m_rows = conn.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY seq ASC",
                (thread_id,),
            ).fetchall()
            f_rows = conn.execute(
                "SELECT * FROM findings WHERE thread_id = ? ORDER BY finding_id ASC",
                (thread_id,),
            ).fetchall()

        target_path = self.threads_dir / t_row["mirror_filename"]
        lines: List[str] = [
            "---",
            f"thread_id: {t_row['thread_id']}",
            f"title: {json.dumps(t_row['title'])}",
            f"status: {t_row['status']}",
            f"created_by: {t_row['created_by']}",
            f"latest_round: {t_row['latest_round']}",
            f"proposal_id: {t_row['proposal_id'] or 'null'}",
            f"updated_at: {t_row['updated_at']}",
            "---",
            "",
            f"# {t_row['thread_id']}: {t_row['title']}",
            "",
            f"**Status**: `{t_row['status']}` | **Round**: `{t_row['latest_round']}` | **Created By**: `{t_row['created_by']}`",
            "",
        ]

        if f_rows:
            lines.extend([
                "## Findings Checklist",
                "",
                "| ID | Sev | Status | File:Lines | Author | Problem | Resolved SHA | Verified By |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for fr in f_rows:
                loc = f"`{fr['file']}:{fr['lines']}`" if fr["file"] else "—"
                prob = str(fr["problem"]).replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| `{fr['finding_id']}` | **{fr['severity']}** | `{fr['status']}` | "
                    f"{loc} | `{fr['author_agent']}` | {prob} | "
                    f"`{fr['resolved_in_sha'] or '-'}` | `{fr['verified_by'] or '-'}` |"
                )
            lines.append("")

        lines.append("## Message Timeline")
        lines.append("")
        for mr in m_rows:
            to_str = ", ".join(json.loads(mr["to_agents_json"])) or "broadcast"
            verdict_str = f" — Verdict: **`{mr['verdict']}`**" if mr["verdict"] else ""
            lines.append(
                f"### [{mr['msg_id']}] R{mr['round_num']} `{mr['from_agent']}` -> `{to_str}` "
                f"(`{mr['intent']}`){verdict_str}"
            )
            lines.append(f"*{mr['created_at']}* — **{mr['subject']}**")
            if mr["code_context_json"]:
                ctx = json.loads(mr["code_context_json"])
                base_part = f"{ctx['base_sha'][:8]}.." if ctx.get("base_sha") else ""
                lines.append(
                    f"- **Code Context**: branch `{ctx.get('branch')}` (`{base_part}{ctx.get('head_sha', '')[:8]}`)"
                )
            adv = json.loads(mr["advisories_json"] or "[]")
            for a in adv:
                lines.append(f"- ⚠️ *{a}*")
            if mr["body"]:
                lines.append("")
                lines.append(mr["body"].strip())
            lines.append("")

        tmp_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp_path, target_path)
        return target_path
