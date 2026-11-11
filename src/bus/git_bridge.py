"""Zero-pollution Git provenance verifier, peer ref mirror, disposable worktree manager, and fleet telemetry."""
from __future__ import annotations
from contextlib import contextmanager
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.bus.schema import WORKSPACE_AGENTS, AgentId, CodeContext, FleetCheckoutStatus
from src.bus.store import BusStore


class CommitNotFoundError(ValueError):
    """Raised when a referenced commit SHA cannot be resolved."""


class UncommittedHandoffError(ValueError):
    """Raised when files claimed in a commit handoff still have uncommitted working-tree modifications."""


class GitBridge:
    """
    Bridges the 5 isolated agent Git checkouts without ever mutating or resetting user checkouts.
    Uses a bare hub mirror (`peer_refs.git`) with Git object `alternates` so peer diffs, behind-main
    counts, and disposable `/tmp` review worktrees work with zero writes to any agent workspace.
    """

    def __init__(
        self,
        workspaces: Dict[str, Dict[str, Any] | Path | str],
        *,
        peer_refs_dir: Path | str,
        worktree_root: Path | str = "/tmp/rl-worktrees",
        golden_agent: AgentId = AgentId.MAIN,
        golden_branch: str = "main",
    ):
        self.workspace_configs: Dict[AgentId, Dict[str, Any]] = {}
        for raw_k, val in workspaces.items():
            try:
                aid = AgentId.normalize(raw_k)
            except ValueError:
                continue
            if isinstance(val, dict):
                path_str = val.get("path", "")
                if not path_str:
                    continue
                # Prefer explicit 'dev' entry over legacy 'gemini' alias if both exist
                if aid in self.workspace_configs and raw_k == "gemini":
                    continue
                self.workspace_configs[aid] = {
                    "name": val.get("name", aid.value.title()),
                    "path": Path(path_str).resolve(),
                    "branch": val.get("branch", aid.value),
                    "role": val.get("role", ""),
                    "binary": val.get("binary"),
                    "model": val.get("model"),
                }
            else:
                self.workspace_configs[aid] = {
                    "name": aid.value.title(),
                    "path": Path(val).resolve(),
                    "branch": aid.value,
                    "role": "",
                }

        self.peer_refs_dir = Path(peer_refs_dir).resolve()
        self.worktree_root = Path(worktree_root).resolve()
        self.golden_agent = golden_agent
        self.golden_branch = golden_branch
        self._ensure_peer_refs_repo()

    @staticmethod
    def _git_ro(repo_path: Path, args: List[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["GIT_OPTIONAL_LOCKS"] = "0"
        return subprocess.run(
            ["git", "-C", str(repo_path), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            check=check,
        )

    @staticmethod
    def _resolve_git_objects_dir(ws_path: Path) -> Optional[Path]:
        dot_git = ws_path / ".git"
        if dot_git.is_dir():
            obj = dot_git / "objects"
            return obj.resolve() if obj.exists() else None
        if dot_git.is_file():
            try:
                txt = dot_git.read_text(encoding="utf-8").strip()
                if txt.startswith("gitdir:"):
                    gitdir = Path(txt.split("gitdir:", 1)[1].strip())
                    if not gitdir.is_absolute():
                        gitdir = (ws_path / gitdir).resolve()
                    commondir_file = gitdir / "commondir"
                    if commondir_file.exists():
                        common = (gitdir / commondir_file.read_text(encoding="utf-8").strip()).resolve()
                        obj = common / "objects"
                        if obj.exists():
                            return obj
                    obj = gitdir / "objects"
                    if obj.exists():
                        return obj
            except OSError:
                return None
        return None

    def _ensure_peer_refs_repo(self) -> None:
        self.peer_refs_dir.parent.mkdir(parents=True, exist_ok=True)
        if not (self.peer_refs_dir / "HEAD").exists():
            subprocess.run(
                ["git", "init", "--bare", str(self.peer_refs_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        self.refresh_alternates()

    def refresh_alternates(self) -> None:
        info_dir = self.peer_refs_dir / "objects" / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        alt_lines: List[str] = []
        seen: set[str] = set()
        for conf in self.workspace_configs.values():
            ws_p: Path = conf["path"]
            if not ws_p.exists():
                continue
            obj_dir = self._resolve_git_objects_dir(ws_p)
            if obj_dir and str(obj_dir) not in seen:
                seen.add(str(obj_dir))
                alt_lines.append(str(obj_dir))
        alt_file = info_dir / "alternates"
        content = ("\n".join(alt_lines) + "\n") if alt_lines else ""
        tmp = info_dir / f".alternates.{os.getpid()}.tmp"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, alt_file)

    def get_workspace_path(self, agent: AgentId | str) -> Optional[Path]:
        aid = AgentId.normalize(agent)
        conf = self.workspace_configs.get(aid)
        return conf["path"] if conf else None

    def resolve_commit(self, repo_path: Path, rev: str) -> str:
        res = self._git_ro(repo_path, ["rev-parse", "--verify", f"{rev}^{{commit}}"], check=False)
        if res.returncode != 0:
            raise CommitNotFoundError(
                f"Commit '{rev}' could not be verified in {repo_path}: {res.stderr.strip()}"
            )
        return res.stdout.strip()

    def verify_commit_handoff(
        self,
        *,
        agent: AgentId | str,
        head_sha: str,
        base_sha: Optional[str] = None,
        workspace_path: Optional[Path | str] = None,
        require_clean_changed_files: bool = True,
    ) -> CodeContext:
        """
        Verifies that `head_sha` (and `base_sha` if given) exist as immutable Git commits, and
        checks that none of the files modified in the commit range currently have uncommitted
        modifications in the sender's working tree (preventing tested-vs-promoted drift).
        Unrelated untracked notes (e.g. `research_notes/draft.md`) are preserved and allowed.
        """
        aid = AgentId.normalize(agent)
        ws = Path(workspace_path).resolve() if workspace_path else self.get_workspace_path(aid)
        if ws is None or not ws.exists():
            raise FileNotFoundError(f"Workspace for agent '{aid.value}' not found: {ws}")

        full_head = self.resolve_commit(ws, head_sha)
        full_base = self.resolve_commit(ws, base_sha) if base_sha else None

        branch_res = self._git_ro(ws, ["rev-parse", "--abbrev-ref", "HEAD"], check=False)
        branch = branch_res.stdout.strip() if branch_res.returncode == 0 else aid.value

        if full_base:
            diff_res = self._git_ro(ws, ["diff", "--name-only", full_base, full_head], check=True)
        else:
            diff_res = self._git_ro(ws, ["show", "--pretty=", "--name-only", full_head], check=True)
        changed_files = [f.strip() for f in diff_res.stdout.splitlines() if f.strip()]

        if require_clean_changed_files and changed_files:
            st_res = self._git_ro(ws, ["status", "--porcelain"], check=False)
            if st_res.returncode == 0 and st_res.stdout.strip():
                dirty_tracked: set[str] = set()
                for line in st_res.stdout.splitlines():
                    if len(line) < 4:
                        continue
                    xy = line[:2]
                    rel_path = line[3:].strip()
                    if " -> " in rel_path:
                        rel_path = rel_path.split(" -> ", 1)[1].strip()
                    if xy != "??":
                        dirty_tracked.add(rel_path)
                dirty_overlap = sorted(set(changed_files) & dirty_tracked)
                if dirty_overlap:
                    raise UncommittedHandoffError(
                        f"Refusing handoff for {full_head[:8]} in {ws}: changed file(s) {dirty_overlap} "
                        f"have uncommitted working-tree edits. Commit or stash them so reviewers see exact code."
                    )

        return CodeContext(
            source_workspace=str(ws),
            branch=branch,
            base_sha=full_base,
            head_sha=full_head,
            changed_files=changed_files,
        )

    def inspect_commit(
        self,
        *,
        from_agent: AgentId | str,
        thread_id: str,
        head_sha: str,
        base_sha: Optional[str] = None,
        source_workspace: Optional[Path | str] = None,
        max_diff_chars: int = 24000,
    ) -> Dict[str, Any]:
        """
        Records `refs/peers/<from_agent>/<thread_id>` inside `peer_refs.git` (zero writes to any
        user checkout) and returns commit metadata, changed files, and bounded unified diff.
        """
        aid = AgentId.normalize(from_agent)
        self.refresh_alternates()
        src_ws = Path(source_workspace).resolve() if source_workspace else self.get_workspace_path(aid)

        # Ensure object is reachable in peer_refs.git (via alternates; fallback to local fetch into bare hub)
        verify = self._git_ro(self.peer_refs_dir, ["rev-parse", "--verify", f"{head_sha}^{{commit}}"], check=False)
        if verify.returncode != 0:
            if src_ws and src_ws.exists():
                fetch_res = subprocess.run(
                    ["git", "-C", str(self.peer_refs_dir), "fetch", str(src_ws), head_sha],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if fetch_res.returncode != 0:
                    raise CommitNotFoundError(
                        f"Failed to fetch commit '{head_sha}' from {src_ws}: {fetch_res.stderr.strip()}"
                    )
                verify = self._git_ro(
                    self.peer_refs_dir, ["rev-parse", "--verify", f"{head_sha}^{{commit}}"], check=False
                )
            if verify.returncode != 0:
                raise CommitNotFoundError(f"Commit '{head_sha}' is not reachable in peer_refs store.")

        full_head = verify.stdout.strip()
        full_base: Optional[str] = None
        if base_sha:
            b_ver = self._git_ro(self.peer_refs_dir, ["rev-parse", "--verify", f"{base_sha}^{{commit}}"], check=False)
            if b_ver.returncode != 0 and src_ws and src_ws.exists():
                subprocess.run(
                    ["git", "-C", str(self.peer_refs_dir), "fetch", str(src_ws), base_sha],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                b_ver = self._git_ro(
                    self.peer_refs_dir, ["rev-parse", "--verify", f"{base_sha}^{{commit}}"], check=False
                )
            if b_ver.returncode != 0:
                raise CommitNotFoundError(f"Base commit '{base_sha}' is not reachable in peer_refs store.")
            full_base = b_ver.stdout.strip()

        peer_ref = f"refs/peers/{aid.value}/{thread_id}"
        subprocess.run(
            ["git", "-C", str(self.peer_refs_dir), "update-ref", peer_ref, full_head],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        subj = self._git_ro(self.peer_refs_dir, ["log", "-1", "--pretty=%s", full_head]).stdout.strip()
        if full_base:
            stat = self._git_ro(self.peer_refs_dir, ["diff", "--stat", full_base, full_head]).stdout.strip()
            names = self._git_ro(self.peer_refs_dir, ["diff", "--name-only", full_base, full_head]).stdout.splitlines()
            patch = self._git_ro(self.peer_refs_dir, ["diff", full_base, full_head]).stdout
        else:
            stat = self._git_ro(self.peer_refs_dir, ["show", "--stat", "--format=", full_head]).stdout.strip()
            names = self._git_ro(
                self.peer_refs_dir, ["show", "--pretty=", "--name-only", full_head]
            ).stdout.splitlines()
            patch = self._git_ro(self.peer_refs_dir, ["show", "--format=", full_head]).stdout

        truncated = len(patch) > max_diff_chars
        if truncated:
            patch = patch[:max_diff_chars] + "\n... [diff truncated; inspect full files inside worktree] ...\n"

        return {
            "peer_ref": peer_ref,
            "head_sha": full_head,
            "base_sha": full_base,
            "subject": subj,
            "changed_files": [n.strip() for n in names if n.strip()],
            "diff_stat": stat,
            "diff_patch": patch,
            "diff_truncated": truncated,
        }

    def create_disposable_worktree(
        self,
        *,
        head_sha: str,
        thread_id: str,
        tag: str = "audit",
        source_workspace: Optional[Path | str] = None,
    ) -> Path:
        """
        Mounts a clean detached worktree under `self.worktree_root` backed by `peer_refs.git`.
        Zero locks or writes occur in any user checkout (`trading*`).
        """
        self.refresh_alternates()
        ver = self._git_ro(self.peer_refs_dir, ["rev-parse", "--verify", f"{head_sha}^{{commit}}"], check=False)
        if ver.returncode != 0 and source_workspace:
            subprocess.run(
                ["git", "-C", str(self.peer_refs_dir), "fetch", str(source_workspace), head_sha],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            ver = self._git_ro(self.peer_refs_dir, ["rev-parse", "--verify", f"{head_sha}^{{commit}}"], check=True)
        elif ver.returncode != 0:
            raise CommitNotFoundError(f"Commit '{head_sha}' not found in peer_refs store.")

        full_head = ver.stdout.strip()
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        wt_name = f"{thread_id}-{tag}-{os.getpid()}-{time.time_ns()}-{full_head[:7]}"
        wt_path = (self.worktree_root / wt_name).resolve()

        subprocess.run(
            ["git", "-C", str(self.peer_refs_dir), "worktree", "add", "--detach", str(wt_path), full_head],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return wt_path

    def cleanup_worktree(self, wt_path: Path | str) -> None:
        wt = Path(wt_path).resolve()
        try:
            subprocess.run(
                ["git", "-C", str(self.peer_refs_dir), "worktree", "remove", "--force", str(wt)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        finally:
            if wt.exists():
                shutil.rmtree(wt, ignore_errors=True)
            subprocess.run(
                ["git", "-C", str(self.peer_refs_dir), "worktree", "prune"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    @contextmanager
    def disposable_worktree(
        self,
        *,
        head_sha: str,
        thread_id: str,
        tag: str = "audit",
        source_workspace: Optional[Path | str] = None,
    ) -> Iterator[Path]:
        wt = self.create_disposable_worktree(
            head_sha=head_sha,
            thread_id=thread_id,
            tag=tag,
            source_workspace=source_workspace,
        )
        try:
            yield wt
        finally:
            self.cleanup_worktree(wt)

    # ------------------------------------------------------------------
    # Fleet Telemetry & Advisory Drift Calculation (Decision #3)
    # ------------------------------------------------------------------
    def get_canonical_main_sha(self, store: Optional[BusStore] = None) -> Optional[str]:
        main_conf = self.workspace_configs.get(self.golden_agent)
        if main_conf and main_conf["path"].exists():
            res = self._git_ro(main_conf["path"], ["rev-parse", "HEAD"], check=False)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        if store:
            b_state = store.get_latest_baseline(self.golden_branch)
            if b_state and b_state.get("head_sha"):
                return str(b_state["head_sha"])
        return None

    def get_agent_drift_advisory(
        self,
        agent: AgentId | str,
        store: Optional[BusStore] = None,
    ) -> tuple[int, int, Optional[str]]:
        """Returns (behind_main, ahead_main, advisory_message_or_none)."""
        aid = AgentId.normalize(agent)
        conf = self.workspace_configs.get(aid)
        if not conf or not conf["path"].exists():
            return 0, 0, None

        main_sha = self.get_canonical_main_sha(store=store)
        if not main_sha or aid == self.golden_agent:
            return 0, 0, None

        self.refresh_alternates()
        head_res = self._git_ro(conf["path"], ["rev-parse", "HEAD"], check=False)
        if head_res.returncode != 0:
            return 0, 0, None
        ws_sha = head_res.stdout.strip()
        if ws_sha == main_sha:
            return 0, 0, None

        cnt_res = self._git_ro(
            self.peer_refs_dir,
            ["rev-list", "--left-right", "--count", f"{main_sha}...{ws_sha}"],
            check=False,
        )
        if cnt_res.returncode != 0:
            return 0, 0, None
        parts = cnt_res.stdout.strip().split()
        if len(parts) != 2:
            return 0, 0, None
        behind, ahead = int(parts[0]), int(parts[1])
        adv = None
        if behind > 0:
            adv = (
                f"Advisory: checkout '{aid.value}' ({conf['branch']}) is {behind} commit(s) behind "
                f"main ({main_sha[:7]}). Consider syncing origin/main if not testing a pinned baseline."
            )
        return behind, ahead, adv

    def get_fleet_telemetry(self, store: Optional[BusStore] = None) -> List[FleetCheckoutStatus]:
        self.refresh_alternates()
        main_sha = self.get_canonical_main_sha(store=store)
        unread_map = store.get_unread_counts() if store else {}
        claims_all = store.list_active_claims() if store else []
        claims_by_agent: Dict[AgentId, List[Any]] = {a: [] for a in WORKSPACE_AGENTS}
        for clm in claims_all:
            if clm.agent in claims_by_agent:
                claims_by_agent[clm.agent].append(clm)

        statuses: List[FleetCheckoutStatus] = []
        for aid in WORKSPACE_AGENTS:
            conf = self.workspace_configs.get(aid)
            if not conf:
                continue
            ws_p: Path = conf["path"]
            exists = ws_p.exists() and ((ws_p / ".git").exists())
            if not exists:
                statuses.append(
                    FleetCheckoutStatus(
                        agent=aid,
                        name=conf["name"],
                        workspace_path=str(ws_p),
                        exists=False,
                        expected_branch=conf["branch"],
                        unread_count=unread_map.get(aid.value, 0),
                    )
                )
                continue

            br_res = self._git_ro(ws_p, ["rev-parse", "--abbrev-ref", "HEAD"], check=False)
            branch = br_res.stdout.strip() if br_res.returncode == 0 else "unknown"

            log_res = self._git_ro(ws_p, ["log", "-1", "--format=%H%x09%ci%x09%s"], check=False)
            head_sha, head_ts, head_subj = "", "", ""
            if log_res.returncode == 0 and log_res.stdout.strip():
                segs = log_res.stdout.strip().split("\t", 2)
                if len(segs) == 3:
                    head_sha, head_ts, head_subj = segs

            st_res = self._git_ro(ws_p, ["status", "--porcelain"], check=False)
            dirty_lines = [ln for ln in st_res.stdout.splitlines() if ln.strip()] if st_res.returncode == 0 else []
            dirty_count = len(dirty_lines)
            is_clean = dirty_count == 0

            behind, ahead = 0, 0
            advisories: List[str] = []
            if branch != conf["branch"]:
                advisories.append(
                    f"Advisory: checkout is on branch '{branch}' (expected '{conf['branch']}')."
                )
            if main_sha and head_sha and aid != self.golden_agent and head_sha != main_sha:
                cnt_res = self._git_ro(
                    self.peer_refs_dir,
                    ["rev-list", "--left-right", "--count", f"{main_sha}...{head_sha}"],
                    check=False,
                )
                if cnt_res.returncode == 0:
                    parts = cnt_res.stdout.strip().split()
                    if len(parts) == 2:
                        behind, ahead = int(parts[0]), int(parts[1])
                        if behind > 0:
                            advisories.append(
                                f"Advisory: {behind} commit(s) behind main ({main_sha[:7]})."
                            )

            statuses.append(
                FleetCheckoutStatus(
                    agent=aid,
                    name=conf["name"],
                    workspace_path=str(ws_p),
                    exists=True,
                    branch=branch,
                    expected_branch=conf["branch"],
                    head_sha=head_sha,
                    head_subject=head_subj,
                    head_timestamp=head_ts,
                    is_clean=is_clean,
                    dirty_count=dirty_count,
                    behind_main=behind,
                    ahead_main=ahead,
                    unread_count=unread_map.get(aid.value, 0),
                    active_claims=claims_by_agent.get(aid, []),
                    advisories=advisories,
                )
            )
        return statuses
