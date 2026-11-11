"""Hybrid cross-checkout knowledge indexer & experiment archiver (Decision #4)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.bus.schema import WORKSPACE_AGENTS, AgentId, Intent, KnowledgeEntry, utc_now_iso
from src.bus.store import BusStore


class KnowledgeManager:
    """
    Maintains the Hybrid Institutional Knowledge Store:
    1. Indexes `research_notes/` and curated `reports/` across all 5 workspaces (`main`, `staging`, `dev`, `claude`, `chatgpt`).
    2. Archives concluded experiment reports (winning, neutral, and failed/unmerged runs) in `research_loop/knowledge/experiments/`
       so findings survive even if a candidate branch is reset or discarded.
    """

    SKIP_REPORT_PATTERNS = (
        "cron_",
        "weekly_validation_status_",
        "data_health",
        "feature_health",
        "fmp_data_",
        "market_regime_status",
    )

    def __init__(
        self,
        store: BusStore,
        workspaces: Optional[Dict[str, Any]] = None,
    ):
        self.store = store
        self.workspaces = workspaces or {}

    @staticmethod
    def _extract_markdown_summary(text: str, fallback_title: str) -> tuple[str, str]:
        title = fallback_title
        summary_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("# ") and title == fallback_title:
                title = stripped[2:].strip()
                continue
            if stripped.startswith("#") or stripped.startswith("---") or stripped.startswith("|"):
                continue
            summary_lines.append(stripped)
            if sum(len(s) for s in summary_lines) >= 360:
                break
        summary = " ".join(summary_lines)[:420] or fallback_title
        return title, summary

    def sync_all_workspaces(self, max_files_per_workspace: int = 120) -> Dict[str, int]:
        """
        Scans `research_notes/` and concluded `reports/*.md` across all configured checkouts,
        plus `knowledge/experiments/*.json`, deduplicating by SHA256 into SQLite FTS5.
        """
        stats = {"scanned": 0, "added": 0, "deduped": 0}

        # 1. Sync local archived experiment JSON manifests
        exp_dir = self.store.knowledge_dir / "experiments"
        if exp_dir.exists():
            for jf in sorted(exp_dir.glob("*.json")):
                try:
                    payload = json.loads(jf.read_text(encoding="utf-8"))
                    stats["scanned"] += 1
                    _, created = self.store.upsert_knowledge(
                        title=payload.get("title", jf.stem),
                        category=payload.get("category", "experiment_report"),
                        source_agent=payload.get("source_agent", "dev"),
                        source_branch=payload.get("source_branch", ""),
                        commit_sha=payload.get("commit_sha"),
                        strategy=payload.get("strategy"),
                        outcome=payload.get("outcome"),
                        summary=payload.get("summary", jf.stem),
                        body_excerpt=payload.get("body_excerpt", ""),
                        tags=payload.get("tags", []),
                        artifact_path=str(jf),
                        content_sha256=payload.get("content_sha256"),
                    )
                    stats["added" if created else "deduped"] += 1
                except Exception:
                    continue

        # 2. Scan all 5 workspaces in deterministic order (claude/dev/chatgpt/staging/main)
        # Put claude & dev first so branch-specific studies get attributed to their author checkout before main merges
        scan_order = [AgentId.CLAUDE, AgentId.DEV, AgentId.CHATGPT, AgentId.STAGING, AgentId.MAIN]
        for aid in scan_order:
            conf = self.workspaces.get(aid.value) or (
                self.workspaces.get("gemini") if aid == AgentId.DEV else None
            )
            if not conf:
                continue
            ws_path = Path(conf["path"] if isinstance(conf, dict) else conf)
            branch = conf.get("branch", aid.value) if isinstance(conf, dict) else aid.value
            if not ws_path.exists():
                continue

            # 2a. Index research_notes/ (markdown files + folder READMEs/syntheses)
            rn_dir = ws_path / "research_notes"
            if rn_dir.exists() and rn_dir.is_dir():
                count = 0
                for sub in sorted(rn_dir.iterdir()):
                    if count >= max_files_per_workspace:
                        break
                    if sub.is_dir():
                        md_files = sorted(sub.rglob("*.md"))
                        if md_files:
                            for mf in md_files[:8]:
                                self._index_markdown_file(
                                    mf,
                                    category="research_note",
                                    agent=aid,
                                    branch=branch,
                                    folder_tag=sub.name,
                                    stats=stats,
                                )
                                count += 1
                        else:
                            # Index study folder metadata even if it contains PDFs/CSVs/scripts
                            child_names = [c.name for c in sorted(sub.iterdir())[:15]]
                            digest = hashlib.sha256(
                                f"research_folder|{sub.name}|{'|'.join(child_names)}".encode("utf-8")
                            ).hexdigest()
                            stats["scanned"] += 1
                            _, created = self.store.upsert_knowledge(
                                title=f"Research Study: {sub.name}",
                                category="research_note",
                                source_agent=aid,
                                source_branch=branch,
                                summary=f"Research study folder '{sub.name}' in {aid.value} ({len(child_names)} artifacts: {', '.join(child_names[:6])}).",
                                body_excerpt="\n".join(child_names),
                                tags=["research_notes", sub.name, aid.value],
                                artifact_path=str(sub),
                                content_sha256=digest,
                            )
                            stats["added" if created else "deduped"] += 1
                            count += 1
                    elif sub.suffix.lower() == ".md":
                        self._index_markdown_file(
                            sub,
                            category="research_note",
                            agent=aid,
                            branch=branch,
                            folder_tag="root",
                            stats=stats,
                        )
                        count += 1

            # 2b. Index curated experiment / audit reports under reports/
            rep_dir = ws_path / "reports"
            if rep_dir.exists() and rep_dir.is_dir():
                md_candidates: List[Path] = []
                try:
                    for item in sorted(rep_dir.iterdir()):
                        if any(item.name.startswith(p) for p in self.SKIP_REPORT_PATTERNS):
                            continue
                        if item.is_file() and item.suffix.lower() == ".md":
                            md_candidates.append(item)
                        elif item.is_dir() and item.name not in (
                            "daily_premarket",
                            "daily_recommendations",
                            "watchlist_performance",
                            "position_supervision",
                            "tail_risk",
                            "history",
                        ):
                            md_candidates.extend(sorted(item.glob("*.md"))[:3])
                except OSError:
                    pass

                for mf in md_candidates[:max_files_per_workspace]:
                    self._index_markdown_file(
                        mf,
                        category="experiment_report",
                        agent=aid,
                        branch=branch,
                        folder_tag=mf.parent.name if mf.parent != rep_dir else "reports",
                        stats=stats,
                    )

        return stats

    def _index_markdown_file(
        self,
        md_path: Path,
        *,
        category: str,
        agent: AgentId,
        branch: str,
        folder_tag: str,
        stats: Dict[str, int],
    ) -> None:
        try:
            raw = md_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return
        if not raw.strip():
            return
        digest = hashlib.sha256(raw.strip().encode("utf-8")).hexdigest()
        fallback = f"{folder_tag} / {md_path.stem}" if folder_tag not in ("root", "reports") else md_path.stem
        title, summary = self._extract_markdown_summary(raw, fallback)
        stats["scanned"] += 1
        _, created = self.store.upsert_knowledge(
            title=title,
            category=category,
            source_agent=agent,
            source_branch=branch,
            summary=summary,
            body_excerpt=raw[:6000],
            tags=[category, folder_tag, md_path.stem],
            artifact_path=str(md_path),
            content_sha256=digest,
        )
        stats["added" if created else "deduped"] += 1

    def publish_experiment_record(
        self,
        *,
        source_agent: AgentId | str,
        title: str,
        summary: str,
        outcome: str,  # POSITIVE | NEGATIVE | NEUTRAL | INVALID
        strategy: str = "standalone_tb",
        source_branch: str = "",
        commit_sha: Optional[str] = None,
        scorecard: Optional[Dict[str, Any]] = None,
        failure_conditions: str = "",
        tags: Optional[Sequence[str]] = None,
        broadcast_on_bus: bool = True,
    ) -> KnowledgeEntry:
        """
        Persists a concluded experiment record (especially negative/unmerged A/Bs) to
        `research_loop/knowledge/experiments/` and indexes it immediately in FTS5.
        """
        aid = AgentId.normalize(source_agent)
        outcome_upper = outcome.strip().upper()
        cat_map = {
            "POSITIVE": "ab_positive",
            "NEGATIVE": "ab_negative",
            "NEUTRAL": "ab_neutral",
        }
        category = cat_map.get(outcome_upper, "experiment_report")
        body_parts = [summary.strip()]
        if failure_conditions:
            body_parts.append(f"Conditions / Failure Mode: {failure_conditions.strip()}")
        if scorecard:
            body_parts.append("Scorecard JSON:\n" + json.dumps(scorecard, indent=2))
        body_text = "\n\n".join(body_parts)
        digest = hashlib.sha256(f"{title}|{commit_sha or ''}|{body_text}".encode("utf-8")).hexdigest()

        exp_dir = self.store.knowledge_dir / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = exp_dir / f"{digest[:12]}-{strategy}-{outcome_upper.lower()}.json"
        record_payload = {
            "title": title,
            "category": category,
            "source_agent": aid.value,
            "source_branch": source_branch or aid.value,
            "commit_sha": commit_sha,
            "strategy": strategy,
            "outcome": outcome_upper,
            "summary": summary,
            "failure_conditions": failure_conditions,
            "scorecard": scorecard,
            "body_excerpt": body_text[:6000],
            "tags": list(tags or [strategy, outcome_upper.lower()]),
            "content_sha256": digest,
            "created_at": utc_now_iso(),
        }
        artifact_file.write_text(json.dumps(record_payload, indent=2), encoding="utf-8")

        entry, _ = self.store.upsert_knowledge(
            title=title,
            category=category,
            source_agent=aid,
            source_branch=source_branch or aid.value,
            commit_sha=commit_sha,
            strategy=strategy,
            outcome=outcome_upper,
            summary=summary,
            body_excerpt=body_text[:6000],
            tags=record_payload["tags"],
            artifact_path=str(artifact_file),
            content_sha256=digest,
        )

        if broadcast_on_bus:
            self.store.send_message(
                from_agent=aid,
                to_agents=[p for p in WORKSPACE_AGENTS if p != aid],
                intent=Intent.KNOWLEDGE_PUBLISHED,
                subject=f"[{outcome_upper}] {title}",
                body=f"{summary}\n\nArtifact: `{artifact_file}`",
                auto_audit=False,
                metadata={"entry_id": entry.entry_id, "outcome": outcome_upper, "strategy": strategy},
            )
        return entry
