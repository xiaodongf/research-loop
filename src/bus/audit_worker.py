"""Headless read-only Auto-Audit Worker executing inside disposable /tmp worktrees."""
from __future__ import annotations
import json
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.agents.claude_runner import ClaudeCodeRunner
from src.bus.capacity_lease import is_pid_alive
from src.bus.git_bridge import GitBridge
from src.bus.schema import (
    AGENT_REVIEW_CONTRACTS,
    EXPLICIT_TARGET_INTENTS,
    READ_ONLY_AUTO_INTENTS,
    AgentId,
    AuditVerdictType,
    CodeContext,
    Finding,
    FindingInput,
    FindingSeverity,
    FindingStatus,
    Intent,
    MessageEnvelope,
    utc_now_iso,
)
from src.bus.store import BusStore


class MutatingIntentForbiddenError(PermissionError):
    """Raised when headless auto-execution is requested for a mutating/non-read-only intent."""


AUDIT_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "summary", "findings", "verified_finding_ids", "reopened_finding_ids"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [
                "APPROVED",
                "APPROVED_WITH_CAUTION",
                "CHANGES_REQUESTED",
                "NEEDS_REVISION",
                "BLOCKED",
                "REJECTED",
            ],
        },
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "category", "file", "lines", "problem", "suggested_fix"],
                "properties": {
                    "severity": {"type": "string", "enum": ["P0", "P1", "P2"]},
                    "category": {"type": "string"},
                    "file": {"type": "string"},
                    "lines": {"type": "string"},
                    "problem": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
            },
        },
        "verified_finding_ids": {"type": "array", "items": {"type": "string"}},
        "reopened_finding_ids": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass
class AuditExecutionRecord:
    reviewer: AgentId
    success: bool
    reply_envelope: Optional[MessageEnvelope] = None
    raw_output: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0
    worktree_used: Optional[str] = None


def format_codex_jsonl_event(line: str) -> Optional[str]:
    """Converts verbose `codex exec --json` JSONL events into concise human-readable progress lines."""
    stripped = line.strip()
    if not stripped:
        return None
    if not stripped.startswith("{"):
        return stripped
    try:
        ev = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped[:240]
    ev_type = ev.get("type", "")
    if ev_type == "thread.started":
        return f"[codex] session started ({ev.get('thread_id', '')[:12]})"
    if ev_type == "turn.started":
        return "[codex] analyzing repository and audit contract..."
    if ev_type in ("item.started", "item.completed"):
        item = ev.get("item") or {}
        itype = item.get("type", "")
        if itype == "command_execution":
            cmd_str = str(item.get("command", "")).strip().replace("\n", " ")
            if ev_type == "item.started":
                return f"[codex] $ {cmd_str[:180]}"
            exit_c = item.get("exit_code")
            return f"[codex] command finished (exit={exit_c})"
        if itype == "reasoning" and ev_type == "item.completed":
            txt = str(item.get("text", "")).strip().replace("\n", " ")
            if txt:
                return f"[codex] thought: {txt[:200]}"
        if itype == "agent_message" and ev_type == "item.completed":
            return "[codex] structured audit verdict generated."
    if ev_type == "error":
        return f"[codex] error: {ev.get('message', stripped[:200])}"
    return None


class AutoAuditWorker:
    """
    Executes strictly read-only headless reviews (`AUDIT_REQUEST`, `PROPOSAL_RFC`, `REMEDIATION_READY`)
    inside disposable detached Git worktrees, with monotonic deadlines and process-group kill (`os.killpg`).
    """

    def __init__(
        self,
        store: BusStore,
        git_bridge: GitBridge,
        *,
        config: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 600.0,
        provider_commands: Optional[Dict[str, str]] = None,
    ):
        self.store = store
        self.git_bridge = git_bridge
        self.config = config or {}
        self.timeout_seconds = timeout_seconds
        self.provider_commands = dict(provider_commands or {})

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.monotonic() + 0.4
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    return
                time.sleep(0.05)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        finally:
            try:
                proc.wait(timeout=2.0)
            except Exception:
                pass

    def run_supervised_command(
        self,
        cmd: List[str],
        *,
        cwd: Path,
        timeout_seconds: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> tuple[int, str, Optional[str]]:
        """
        Spawns `cmd` in its own POSIX process group (`start_new_session=True`) with `stdin=DEVNULL`
        (crucial so non-TTY callers like Streamlit never cause `codex exec` or `claude -p` to block
        waiting for piped stdin EOF), streams output live, and kills the entire process group on timeout.
        """
        eff_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        deadline = time.monotonic() + eff_timeout
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env or os.environ.copy(),
            start_new_session=True,
        )

        chunks: List[str] = []
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        partial = ""

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_process_group(proc)
                    return -1, "".join(chunks), f"Audit worker timed out after {eff_timeout:.1f}s (process group killed)"

                rlist, _, _ = select.select([fd], [], [], min(0.1, remaining))
                if rlist:
                    try:
                        data = os.read(fd, 8192).decode("utf-8", errors="replace")
                    except BlockingIOError:
                        data = ""
                    if data:
                        chunks.append(data)
                        if on_output:
                            combined = partial + data
                            lines = combined.split("\n")
                            partial = lines.pop()
                            for ln in lines:
                                on_output(ln)
                    elif proc.poll() is not None:
                        break
                elif proc.poll() is not None:
                    while True:
                        try:
                            tail = os.read(fd, 8192).decode("utf-8", errors="replace")
                        except BlockingIOError:
                            tail = ""
                        if not tail:
                            break
                        chunks.append(tail)
                    break

            if partial and on_output:
                on_output(partial)
            exit_code = proc.wait(timeout=2.0)
            full_text = "".join(chunks)
            err = None if exit_code == 0 else f"Process exited with code {exit_code}"
            return exit_code, full_text, err
        except Exception as exc:
            self._kill_process_group(proc)
            return -1, "".join(chunks), str(exc)

    def build_audit_prompt(
        self,
        *,
        reviewer: AgentId,
        request_msg: MessageEnvelope,
        thread_bundle: Dict[str, Any],
        diff_info: Optional[Dict[str, Any]] = None,
        mounted_branch_note: str = "",
    ) -> str:
        contract = AGENT_REVIEW_CONTRACTS.get(
            reviewer,
            {"title": "Quantitative Peer Auditor", "focus": "Audit correctness, causality, and parity."},
        )
        findings_list: List[Finding] = thread_bundle.get("findings", [])
        open_findings_block = ""
        if findings_list:
            rows_txt = [
                f"- {f.id} [{f.severity.value} | {f.status.value}] ({f.file}:{f.lines}): {f.problem} "
                f"(resolved_in={f.resolved_in_sha or 'none'})"
                for f in findings_list
            ]
            open_findings_block = (
                "\nEXISTING THREAD FINDINGS TO VERIFY / TRACK:\n"
                + "\n".join(rows_txt)
                + "\nFor each FIXED_PENDING_VERIFY finding, include its ID in `verified_finding_ids` if fixed, "
                "or in `reopened_finding_ids` if still broken.\n"
            )

        diff_block = ""
        if diff_info:
            diff_block = (
                f"\nGIT COMMIT UNDER REVIEW:\n"
                f"- Head SHA: {diff_info['head_sha']}\n"
                f"- Base SHA: {diff_info.get('base_sha') or 'parent'}\n"
                f"- Subject: {diff_info.get('subject', '')}\n"
                f"- Changed Files: {', '.join(diff_info.get('changed_files', []))}\n\n"
                f"DIFF STAT:\n{diff_info.get('diff_stat', '')}\n\n"
                f"DIFF PATCH:\n{diff_info.get('diff_patch', '')}\n"
            )

        ws_note = f"\nWORKTREE CONTEXT: {mounted_branch_note}\n" if mounted_branch_note else ""

        return (
            f"You are `{reviewer.value}` acting in your specialized role: **{contract['title']}**.\n"
            f"You are running inside a READ-ONLY disposable Git worktree containing the full repository. "
            f"DO NOT modify or commit files.\n"
            f"{ws_note}\n"
            f"SPECIALIZED AUDIT CONTRACT:\n{contract['focus']}\n\n"
            f"THREAD: {request_msg.thread_id} (Round {request_msg.round_num})\n"
            f"INTENT: {request_msg.intent.value} from `{request_msg.from_agent.value}`\n"
            f"SUBJECT: {request_msg.subject}\n\n"
            f"REQUEST BODY:\n{request_msg.body}\n"
            f"{open_findings_block}"
            f"{diff_block}\n"
            f"OUTPUT CONTRACT (STRICT JSON ONLY):\n"
            f"Respond with a single valid JSON object matching:\n"
            f'{{"verdict": "APPROVED|APPROVED_WITH_CAUTION|CHANGES_REQUESTED|NEEDS_REVISION|BLOCKED|REJECTED", '
            f'"summary": "concise executive summary", '
            f'"findings": [{{"severity": "P0|P1|P2", "category": "...", "file": "...", "lines": "...", '
            f'"problem": "...", "suggested_fix": "..."}}], '
            f'"verified_finding_ids": ["THR-0001-F1"], "reopened_finding_ids": []}}\n'
        )

    @staticmethod
    def parse_audit_json(raw_text: str) -> Dict[str, Any]:
        """Extracts structured verdict JSON from raw CLI / LLM output (handling Claude wrapper JSON or fenced blocks)."""
        text = raw_text.strip()
        if not text:
            raise ValueError("Empty audit output from provider.")

        try:
            outer = json.loads(text)
            if isinstance(outer, dict):
                if "verdict" in outer and "summary" in outer:
                    return outer
                if "result" in outer and isinstance(outer["result"], str):
                    return AutoAuditWorker.parse_audit_json(outer["result"])
        except json.JSONDecodeError:
            pass

        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        for cand in reversed(fenced):
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict) and "verdict" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            cand = text[first : last + 1]
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed

        raise ValueError(f"Could not parse valid audit verdict JSON from output: {text[:400]}")

    def _invoke_provider_readonly(
        self,
        *,
        reviewer: AgentId,
        prompt: str,
        worktree_dir: Path,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> tuple[int, str, Optional[str]]:
        # 1. Explicit test/custom command override (`mock:...` or shell template)
        override = self.provider_commands.get(reviewer.value)
        if override:
            raw_cmd = override[len("mock:") :] if override.startswith("mock:") else override
            env = os.environ.copy()
            env["RL_AUDIT_PROMPT"] = prompt
            env["RL_AUDIT_REVIEWER"] = reviewer.value
            return self.run_supervised_command(
                ["bash", "-c", raw_cmd],
                cwd=worktree_dir,
                env=env,
                on_output=on_output,
            )

        ws_confs = self.config.get("workspaces", {})

        # 2. ChatGPT / OpenAI Codex CLI (`codex exec --ephemeral -s read-only --json`)
        if reviewer == AgentId.CHATGPT:
            codex_bin = (
                ws_confs.get("chatgpt", {}).get("binary")
                or "/usr/lib/chatgpt/resources/codex"
            )
            with tempfile.TemporaryDirectory(prefix="rl-codex-schema-") as td:
                schema_file = Path(td) / "audit_schema.json"
                out_file = Path(td) / "last_msg.json"
                schema_file.write_text(json.dumps(AUDIT_OUTPUT_JSON_SCHEMA), encoding="utf-8")
                cmd = [
                    str(codex_bin),
                    "exec",
                    "--ephemeral",
                    "-s",
                    "read-only",
                    "--skip-git-repo-check",
                    "--color",
                    "never",
                    "--json",
                    "--output-schema",
                    str(schema_file),
                    "-o",
                    str(out_file),
                    "-C",
                    str(worktree_dir),
                    prompt,
                ]
                env = os.environ.copy()
                res_dir = "/usr/lib/chatgpt/resources"
                if Path(res_dir).exists() and res_dir not in env.get("PATH", ""):
                    env["PATH"] = f"{res_dir}:{env.get('PATH', '')}"

                def _codex_stream_filter(raw_ln: str) -> None:
                    if not on_output:
                        return
                    formatted = format_codex_jsonl_event(raw_ln)
                    if formatted:
                        on_output(formatted)

                code, stdout_txt, err = self.run_supervised_command(
                    cmd, cwd=worktree_dir, env=env, on_output=_codex_stream_filter
                )
                if code == 0 and out_file.exists():
                    return 0, out_file.read_text(encoding="utf-8"), None
                return code, stdout_txt, err

        # 3. Claude Code CLI (`claude -p --output-format json --allowedTools Read,Grep,Glob`)
        if reviewer == AgentId.CLAUDE:
            conf_bin = ws_confs.get("claude", {}).get("binary")
            claude_bin = ClaudeCodeRunner.resolve_claude_binary(conf_bin)
            if on_output:
                on_output(f"[claude] running read-only audit via {claude_bin} in {worktree_dir}...")
            cmd = [
                claude_bin,
                "-p",
                prompt,
                "--output-format",
                "json",
                "--allowedTools",
                "Read,Grep,Glob",
            ]
            return self.run_supervised_command(cmd, cwd=worktree_dir, on_output=on_output)

        # 4. Gemini / Dev (or Main/Staging fallback via Gemini API)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            model_name = ws_confs.get("dev", ws_confs.get("gemini", {})).get("model", "gemini-3.8-flash")
            if on_output:
                on_output(f"[gemini] invoking {model_name} read-only audit...")
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(model=model_name, contents=prompt)
                if resp and resp.text:
                    return 0, resp.text, None
            except Exception as exc:
                return -1, "", f"Gemini API audit failed: {exc}"

        return -1, "", f"No configured read-only audit provider available for '{reviewer.value}'."

    def _resolve_effective_code_context(self, env_msg: MessageEnvelope) -> tuple[Optional[CodeContext], str]:
        """
        Returns `(effective_code_context, mounted_branch_note)`.
        If the message already carries `code_context.head_sha`, uses it directly.
        Otherwise automatically mounts `HEAD` (with `HEAD~1` diff) of `metadata.target_checkout`
        or `golden_agent` (`main`), so auditors never land in an empty folder.
        """
        if env_msg.code_context and env_msg.code_context.head_sha:
            ctx = env_msg.code_context
            return (
                ctx,
                f"Mounted committed snapshot `{ctx.head_sha[:8]}` from branch `{ctx.branch}` ({ctx.source_workspace}).",
            )

        meta_target = env_msg.metadata.get("target_checkout")
        if meta_target == "none":
            return None, ""

        fallback_agent = self.git_bridge.golden_agent
        if meta_target:
            try:
                fallback_agent = AgentId.normalize(meta_target)
            except ValueError:
                pass
        elif env_msg.from_agent != AgentId.HUMAN:
            lower_txt = f"{env_msg.subject} {env_msg.body}".lower()
            if "main branch" in lower_txt or "on main" in lower_txt:
                fallback_agent = self.git_bridge.golden_agent
            else:
                ws_candidate = self.git_bridge.get_workspace_path(env_msg.from_agent)
                if ws_candidate and ws_candidate.exists():
                    fallback_agent = env_msg.from_agent

        ws_path = self.git_bridge.get_workspace_path(fallback_agent)
        if ws_path and ws_path.exists():
            try:
                head_sha = self.git_bridge.resolve_commit(ws_path, "HEAD")
                try:
                    base_sha: Optional[str] = self.git_bridge.resolve_commit(ws_path, "HEAD~1")
                except Exception:
                    base_sha = None
                ctx = self.git_bridge.verify_commit_handoff(
                    agent=fallback_agent,
                    head_sha=head_sha,
                    base_sha=base_sha,
                    workspace_path=ws_path,
                    require_clean_changed_files=False,
                )
                return (
                    ctx,
                    f"Auto-mounted `{fallback_agent.value}` (`{ctx.branch}`) HEAD `{head_sha[:8]}` from `{ws_path}`.",
                )
            except Exception:
                return None, ""
        return None, ""

    def execute_audit_for_message(
        self,
        message: MessageEnvelope | str,
        *,
        reviewers: Optional[List[AgentId | str]] = None,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> List[AuditExecutionRecord]:
        """
        Executes read-only headless audit(s) for `message` across explicit target `reviewers`
        (defaults to `message.to_agents`). Refuses mutating intents unconditionally.
        """
        env_msg = self.store.get_message(message) if isinstance(message, str) else message
        if env_msg is None:
            raise KeyError(f"Message '{message}' not found in BusStore.")

        if env_msg.intent not in READ_ONLY_AUTO_INTENTS:
            raise MutatingIntentForbiddenError(
                f"AutoAuditWorker refuses intent '{env_msg.intent.value}': headless auto-execution is "
                f"strictly restricted to read-only intents {[i.value for i in sorted(READ_ONLY_AUTO_INTENTS, key=lambda x: x.value)]}."
            )

        target_reviewers = (
            [AgentId.normalize(r) for r in reviewers]
            if reviewers is not None
            else [r for r in env_msg.to_agents if r != AgentId.HUMAN]
        )
        if env_msg.intent in EXPLICIT_TARGET_INTENTS and not target_reviewers:
            raise ValueError("Cannot run audit without at least one explicit target reviewer.")

        self.store.mark_auto_dispatched(env_msg.msg_id)
        thread_bundle = self.store.get_thread(env_msg.thread_id) or {"findings": []}

        eff_ctx, mounted_note = self._resolve_effective_code_context(env_msg)
        diff_info: Optional[Dict[str, Any]] = None
        if eff_ctx and eff_ctx.head_sha:
            try:
                diff_info = self.git_bridge.inspect_commit(
                    from_agent=env_msg.from_agent if env_msg.from_agent != AgentId.HUMAN else self.git_bridge.golden_agent,
                    thread_id=env_msg.thread_id,
                    head_sha=eff_ctx.head_sha,
                    base_sha=eff_ctx.base_sha,
                    source_workspace=eff_ctx.source_workspace,
                )
            except Exception as exc:
                if on_output:
                    on_output(f"[git] warning inspecting commit diff: {exc}")

        results: List[AuditExecutionRecord] = []
        for reviewer in target_reviewers:
            t0 = time.monotonic()
            prompt = self.build_audit_prompt(
                reviewer=reviewer,
                request_msg=env_msg,
                thread_bundle=thread_bundle,
                diff_info=diff_info,
                mounted_branch_note=mounted_note,
            )

            wt_cm = (
                self.git_bridge.disposable_worktree(
                    head_sha=eff_ctx.head_sha,
                    thread_id=env_msg.thread_id,
                    tag=f"{reviewer.value}-r{env_msg.round_num}",
                    source_workspace=eff_ctx.source_workspace,
                )
                if (eff_ctx and eff_ctx.head_sha)
                else tempfile.TemporaryDirectory(prefix=f"rl-rfc-{env_msg.thread_id}-")
            )

            with wt_cm as wt_raw:
                wt_path = Path(wt_raw).resolve()
                if on_output:
                    on_output(
                        f"[worker] starting `{reviewer.value}` read-only audit on {env_msg.thread_id} "
                        f"in worktree {wt_path}"
                    )
                exit_code, raw_out, err_msg = self._invoke_provider_readonly(
                    reviewer=reviewer,
                    prompt=prompt,
                    worktree_dir=wt_path,
                    on_output=on_output,
                )
                dur = time.monotonic() - t0
                if exit_code != 0:
                    err_detail = err_msg or f"Non-zero exit code {exit_code}"
                    self.store.send_message(
                        from_agent=reviewer,
                        to_agents=[env_msg.from_agent],
                        intent=Intent.ADVISORY_NOTE,
                        thread_id=env_msg.thread_id,
                        in_reply_to=env_msg.msg_id,
                        subject=f"[AUDIT_ERROR] {reviewer.value} failed after {dur:.1f}s",
                        body=f"Headless audit failed: `{err_detail}`\n\n```\n{raw_out[-1200:]}\n```",
                        auto_audit=False,
                    )
                    results.append(
                        AuditExecutionRecord(
                            reviewer=reviewer,
                            success=False,
                            raw_output=raw_out,
                            error=err_detail,
                            duration_seconds=dur,
                            worktree_used=str(wt_path),
                        )
                    )
                    continue

                try:
                    parsed = self.parse_audit_json(raw_out)
                    verdict = AuditVerdictType(parsed["verdict"])
                    summary = str(parsed.get("summary", "")).strip()
                    raw_findings = parsed.get("findings", []) or []
                    verified_ids = [str(fid) for fid in (parsed.get("verified_finding_ids") or [])]
                    reopened_ids = [str(fid) for fid in (parsed.get("reopened_finding_ids") or [])]

                    for rfid in reopened_ids:
                        self.store.update_finding_status(rfid, FindingStatus.OPEN)

                    finding_inputs = [
                        FindingInput(
                            severity=FindingSeverity(f.get("severity", "P1")),
                            category=str(f.get("category", "audit")),
                            file=str(f.get("file", "")),
                            lines=str(f.get("lines", "")),
                            problem=str(f["problem"]),
                            suggested_fix=str(f.get("suggested_fix", "")),
                        )
                        for f in raw_findings
                    ]

                    reply_intent = (
                        Intent.PEER_CRITIQUE if env_msg.intent == Intent.PROPOSAL_RFC else Intent.AUDIT_VERDICT
                    )
                    reply_env = self.store.send_message(
                        from_agent=reviewer,
                        to_agents=[env_msg.from_agent],
                        cc_agents=env_msg.cc_agents,
                        intent=reply_intent,
                        thread_id=env_msg.thread_id,
                        in_reply_to=env_msg.msg_id,
                        subject=f"Re: {env_msg.subject} [{verdict.value}]",
                        body=summary,
                        code_context=eff_ctx,
                        verdict=verdict,
                        new_findings=finding_inputs,
                        verified_findings=verified_ids,
                        auto_audit=False,
                    )
                    results.append(
                        AuditExecutionRecord(
                            reviewer=reviewer,
                            success=True,
                            reply_envelope=reply_env,
                            raw_output=raw_out,
                            duration_seconds=dur,
                            worktree_used=str(wt_path),
                        )
                    )
                except Exception as exc:
                    err_detail = f"Failed to parse structured audit JSON: {exc}"
                    self.store.send_message(
                        from_agent=reviewer,
                        to_agents=[env_msg.from_agent],
                        intent=Intent.ADVISORY_NOTE,
                        thread_id=env_msg.thread_id,
                        in_reply_to=env_msg.msg_id,
                        subject=f"[AUDIT_ERROR] {reviewer.value} output parse failure",
                        body=f"`{err_detail}`\n\n```\n{raw_out[-1200:]}\n```",
                        auto_audit=False,
                    )
                    results.append(
                        AuditExecutionRecord(
                            reviewer=reviewer,
                            success=False,
                            raw_output=raw_out,
                            error=err_detail,
                            duration_seconds=dur,
                            worktree_used=str(wt_path),
                        )
                    )

        return results

    def poll_and_execute_once(
        self,
        *,
        limit: int = 5,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> List[AuditExecutionRecord]:
        claimed = self.store.claim_pending_auto_audits(limit=limit)
        all_records: List[AuditExecutionRecord] = []
        for msg in claimed:
            all_records.extend(self.execute_audit_for_message(msg, on_output=on_output))
        return all_records


# ----------------------------------------------------------------------
# Non-Blocking Background Audit Job Launcher & Live Telemetry Helpers
# ----------------------------------------------------------------------
def write_audit_job_meta(runs_dir: Path, msg_id: str, payload: Dict[str, Any]) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    meta_file = runs_dir / f"{msg_id}.json"
    tmp_file = runs_dir / f".{msg_id}.{os.getpid()}.tmp"
    tmp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_file, meta_file)
    return meta_file


def launch_background_audit(
    *,
    msg_id: str,
    thread_id: str,
    reviewers: List[str],
    root_dir: Path,
    state_dir: Path,
    config_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Spawns `scripts/cli.py bus audit-run --msg-id <msg_id>` in a detached background session
    (`stdin=DEVNULL`, `start_new_session=True`), streaming live output to `state/audit_runs/<msg_id>.log`
    so the Streamlit UI responds in <0.1s and tails live progress without blocking!
    """
    runs_dir = state_dir / "audit_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / f"{msg_id}.log"

    cli_script = root_dir / "scripts" / "cli.py"
    cmd = [sys.executable, "-u", str(cli_script)]
    if config_path:
        cmd.extend(["--config", str(config_path)])
    if db_path:
        cmd.extend(["--db", str(db_path)])
    cmd.extend(["bus", "audit-run", "--msg-id", msg_id])

    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root_dir),
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        log_fh.close()

    meta = {
        "msg_id": msg_id,
        "thread_id": thread_id,
        "reviewers": reviewers,
        "pid": proc.pid,
        "status": "RUNNING",
        "started_at": utc_now_iso(),
        "started_epoch": time.time(),
        "finished_at": None,
        "log_path": str(log_path),
        "results": [],
    }
    write_audit_job_meta(runs_dir, msg_id, meta)
    return meta


def list_audit_jobs(state_dir: Path, limit: int = 8) -> List[Dict[str, Any]]:
    runs_dir = state_dir / "audit_runs"
    if not runs_dir.exists():
        return []
    items: List[Dict[str, Any]] = []
    for jf in sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "RUNNING":
            pid = int(data.get("pid") or -1)
            if not is_pid_alive(pid):
                data["status"] = "FINISHED"
                write_audit_job_meta(runs_dir, data["msg_id"], data)
        log_p = Path(data.get("log_path", ""))
        if log_p.exists():
            try:
                lines = log_p.read_text(encoding="utf-8", errors="replace").splitlines()
                data["log_tail"] = "\n".join(lines[-25:])
            except OSError:
                data["log_tail"] = ""
        else:
            data["log_tail"] = ""
        items.append(data)
    return items


def stop_audit_job(state_dir: Path, msg_id: str) -> bool:
    runs_dir = state_dir / "audit_runs"
    meta_file = runs_dir / f"{msg_id}.json"
    if not meta_file.exists():
        return False
    data = json.loads(meta_file.read_text(encoding="utf-8"))
    pid = int(data.get("pid") or -1)
    if pid > 0 and is_pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    data["status"] = "CANCELLED"
    data["finished_at"] = utc_now_iso()
    write_audit_job_meta(runs_dir, msg_id, data)
    return True
