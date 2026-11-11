"""Atomic cross-process host capacity lease manager wrapping check_host_capacity.py."""
from __future__ import annotations
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.bus.schema import AgentId, EvalLease, LeaseAcquisitionResult, utc_now_iso
from src.bus.store import BusStore


def is_pid_alive(pid: int) -> bool:
    """Checks whether an OS PID exists and is not a zombie ('Z') process."""
    if pid <= 0:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            txt = stat_path.read_text(encoding="utf-8")
            after_comm = txt[txt.rindex(")") + 2 :].split()
            if after_comm and after_comm[0] == "Z":
                return False
            return True
        except (OSError, ValueError):
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class CapacityLeaseManager:
    """
    Coordinates evaluation / A/B launches across all 5 workspaces using an atomic SQLite
    `BEGIN IMMEDIATE` lease table combined with live `/proc` supply from `check_host_capacity.py`.
    Eliminates the simultaneous-launch TOCTOU race while preserving advisory warnings and
    explicit `ignore_host_guard` overrides.
    """

    def __init__(
        self,
        store: BusStore,
        *,
        capacity_script_path: Optional[Path | str] = "/home/xiaodong/workspace/trading-exp/scripts/check_host_capacity.py",
        host_probe: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        self.store = store
        self.capacity_script_path = Path(capacity_script_path) if capacity_script_path else None
        self.host_probe = host_probe
        self._loaded_module: Optional[Any] = None

    def _get_capacity_module(self) -> Optional[Any]:
        if self._loaded_module is not None:
            return self._loaded_module
        if self.capacity_script_path and self.capacity_script_path.exists():
            try:
                spec = importlib.util.spec_from_file_location("host_capacity_mod", str(self.capacity_script_path))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self._loaded_module = mod
                    return mod
            except Exception:
                return None
        return None

    def _probe_host_supply(self) -> Dict[str, Any]:
        if self.host_probe is not None:
            return dict(self.host_probe())

        mod = self._get_capacity_module()
        if mod is not None and hasattr(mod, "check_capacity"):
            raw = mod.check_capacity(exclude={os.getpid()})
            return {
                "ncpu": int(raw.get("ncpu", os.cpu_count() or 1)),
                "load1": float(raw.get("load1", 0.0)),
                "load5": float(raw.get("load5", 0.0)),
                "avail_mem_gib": float(raw.get("avail_mem_gib", 64.0)),
                "running_jobs": list(raw.get("running_jobs", [])),
            }

        ncpu = os.cpu_count() or 1
        load1, load5, _ = os.getloadavg()
        avail_mem = 64.0
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    avail_mem = int(line.split()[1]) / 1048576.0
                    break
        return {
            "ncpu": ncpu,
            "load1": round(load1, 2),
            "load5": round(load5, 2),
            "avail_mem_gib": round(avail_mem, 2),
            "running_jobs": [],
        }

    @staticmethod
    def read_ab_partial_progress(checkpoint_dir: Optional[str]) -> Optional[Dict[str, int]]:
        if not checkpoint_dir:
            return None
        cp_path = Path(checkpoint_dir)
        if not cp_path.exists() or not cp_path.is_dir():
            return None
        completed = 0
        failed = 0
        for jf in cp_path.glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if data.get("status") == "failed" or data.get("error"):
                    failed += 1
                else:
                    completed += 1
            except Exception:
                continue
        if completed == 0 and failed == 0:
            return None
        return {"completed_folds": completed, "failed_folds": failed}

    @staticmethod
    def _row_to_lease(r: Any) -> EvalLease:
        return EvalLease(
            lease_id=r["lease_id"],
            agent=AgentId.normalize(r["agent"]),
            pid=int(r["pid"]),
            workspace_path=r["workspace_path"],
            strategy=r["strategy"],
            mode=r["mode"],
            workers=float(r["workers"]),
            threads_per_worker=float(r["threads_per_worker"]),
            mem_per_worker_gib=float(r["mem_per_worker_gib"]),
            thread_id=r["thread_id"],
            checkpoint_dir=r["checkpoint_dir"],
            acquired_at=r["acquired_at"],
            released_at=r["released_at"],
            status=r["status"],
        )

    def _reclaim_stale_leases_tx(self, conn: Any, now_iso: str) -> List[str]:
        rows = conn.execute("SELECT * FROM eval_leases WHERE status = 'ACTIVE'").fetchall()
        reclaimed: List[str] = []
        for r in rows:
            if not is_pid_alive(int(r["pid"])):
                conn.execute(
                    "UPDATE eval_leases SET status = 'EXPIRED_DEAD_PID', released_at = ? WHERE lease_id = ?",
                    (now_iso, r["lease_id"]),
                )
                reclaimed.append(r["lease_id"])
        if reclaimed:
            self._notify_next_waiter_tx(conn, now_iso)
        return reclaimed

    @staticmethod
    def _notify_next_waiter_tx(conn: Any, now_iso: str) -> Optional[str]:
        nxt = conn.execute(
            "SELECT * FROM eval_waitlist WHERE status = 'WAITING' ORDER BY requested_at ASC LIMIT 1"
        ).fetchone()
        if nxt:
            conn.execute(
                "UPDATE eval_waitlist SET status = 'NOTIFIED', notified_at = ? WHERE wait_id = ?",
                (now_iso, nxt["wait_id"]),
            )
            return str(nxt["wait_id"])
        return None

    def acquire_eval_lease(
        self,
        *,
        agent: AgentId | str,
        pid: Optional[int] = None,
        workspace_path: str = "",
        strategy: str = "standalone_tb",
        mode: str = "quick",
        workers: float = 4.0,
        threads_per_worker: float = 6.0,
        mem_per_worker_gib: float = 8.0,
        reserve_cores: float = 4.0,
        reserve_mem_gib: float = 16.0,
        memory_tolerance_pct: float = 20.0,
        ramp_seconds: float = 300.0,
        thread_id: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        enqueue_if_busy: bool = True,
        ignore_host_guard: bool = False,
        advisories: Optional[List[str]] = None,
    ) -> LeaseAcquisitionResult:
        if not math.isfinite(memory_tolerance_pct) or not (0.0 <= memory_tolerance_pct <= 20.0):
            raise ValueError("memory_tolerance_pct must be between 0 and 20")

        aid = AgentId.normalize(agent)
        eff_pid = int(pid if pid is not None else os.getpid())
        now_iso = utc_now_iso()
        now_epoch = time.time()
        adv_list = list(advisories or [])

        # Probe host supply, then serialize decision + reservation inside BEGIN IMMEDIATE
        supply = self._probe_host_supply()

        with self.store.immediate_tx() as conn:
            self._reclaim_stale_leases_tx(conn, now_iso)
            active_rows = conn.execute(
                "SELECT * FROM eval_leases WHERE status = 'ACTIVE' ORDER BY acquired_epoch ASC"
            ).fetchall()
            active_leases = [self._row_to_lease(r) for r in active_rows]

            ncpu = int(supply["ncpu"])
            load1 = float(supply["load1"])
            load5 = float(supply["load5"])
            avail_mem = float(supply["avail_mem_gib"])
            proc_jobs = supply.get("running_jobs", [])

            # Combine ramping CPU & unclaimed memory from active leases + /proc jobs without double-counting PIDs
            leased_pids = {int(r["pid"]) for r in active_rows}
            ramping_cores = 0.0
            ramping_mem_gib = 0.0

            for r in active_rows:
                age = max(0.0, now_epoch - float(r["acquired_epoch"]))
                if age < ramp_seconds:
                    ramping_cores += float(r["workers"]) * float(r["threads_per_worker"])
                    ramping_mem_gib += float(r["workers"]) * float(r["mem_per_worker_gib"])

            for pj in proc_jobs:
                pj_pid = int(pj.get("pid", -1))
                pj_age = pj.get("age_s")
                if pj_pid not in leased_pids and pj_age is not None and float(pj_age) < ramp_seconds:
                    ramping_cores += workers * threads_per_worker
                    ramping_mem_gib += workers * mem_per_worker_gib

            need_cores = workers * threads_per_worker
            need_mem = workers * mem_per_worker_gib

            effective_idle_cores = ncpu - max(load1, load5) - ramping_cores
            effective_avail_mem = max(0.0, avail_mem - ramping_mem_gib)

            ok_cpu = effective_idle_cores >= (need_cores + reserve_cores)
            estimated_total_mem = need_mem + reserve_mem_gib
            memory_limit = effective_avail_mem * (1.0 + memory_tolerance_pct / 100.0)
            ok_mem = estimated_total_mem <= memory_limit
            ok = bool(ok_cpu and ok_mem)

            snapshot = {
                "ok": ok,
                "ok_cpu": ok_cpu,
                "ok_mem": ok_mem,
                "ncpu": ncpu,
                "load1": load1,
                "load5": load5,
                "idle_cores": round(effective_idle_cores, 1),
                "need_cores": need_cores,
                "reserve_cores": reserve_cores,
                "avail_mem_gib": round(effective_avail_mem, 2),
                "raw_avail_mem_gib": round(avail_mem, 2),
                "need_mem_gib": need_mem,
                "reserve_mem_gib": reserve_mem_gib,
                "memory_limit_gib": round(memory_limit, 2),
                "active_lease_count": len(active_leases),
                "proc_eval_jobs": proc_jobs,
            }

            if ok or ignore_host_guard:
                if ignore_host_guard and not ok:
                    adv_list.append("Advisory: host capacity guard bypassed via ignore_host_guard=True.")
                l_num = self.store._next_counter(conn, "lease")
                lid = f"LSE-{l_num:05d}"
                conn.execute(
                    """
                    INSERT INTO eval_leases(
                        lease_id, agent, pid, workspace_path, strategy, mode,
                        workers, threads_per_worker, mem_per_worker_gib,
                        thread_id, checkpoint_dir, acquired_at, acquired_epoch, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
                    """,
                    (
                        lid,
                        aid.value,
                        eff_pid,
                        workspace_path,
                        strategy,
                        mode,
                        workers,
                        threads_per_worker,
                        mem_per_worker_gib,
                        thread_id,
                        checkpoint_dir,
                        now_iso,
                        now_epoch,
                    ),
                )
                # Mark any waiting entry for this agent/thread as fulfilled
                conn.execute(
                    "UPDATE eval_waitlist SET status = 'FULFILLED' WHERE agent = ? AND status IN ('WAITING', 'NOTIFIED')",
                    (aid.value,),
                )
                lease = EvalLease(
                    lease_id=lid,
                    agent=aid,
                    pid=eff_pid,
                    workspace_path=workspace_path,
                    strategy=strategy,
                    mode=mode,
                    workers=workers,
                    threads_per_worker=threads_per_worker,
                    mem_per_worker_gib=mem_per_worker_gib,
                    thread_id=thread_id,
                    checkpoint_dir=checkpoint_dir,
                    acquired_at=now_iso,
                    status="ACTIVE",
                )
                return LeaseAcquisitionResult(
                    acquired=True,
                    lease=lease,
                    active_holders=[*active_leases, lease],
                    capacity_snapshot=snapshot,
                    advisories=adv_list,
                    reason="OK to launch",
                )

            wait_id = None
            wait_pos = None
            if enqueue_if_busy:
                w_num = self.store._next_counter(conn, "waitlist")
                wait_id = f"WAI-{w_num:05d}"
                conn.execute(
                    """
                    INSERT INTO eval_waitlist(wait_id, agent, pid, strategy, mode, workers, thread_id, requested_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'WAITING')
                    """,
                    (wait_id, aid.value, eff_pid, strategy, mode, workers, thread_id, now_iso),
                )
                cnt_row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM eval_waitlist WHERE status = 'WAITING'"
                ).fetchone()
                wait_pos = int(cnt_row["cnt"])

            holder_desc = ", ".join(f"{h.agent.value}(PID {h.pid}, {h.strategy}/{h.mode})" for h in active_leases) or "OS load"
            reason = (
                f"WAIT (insufficient host capacity: idle_cores={snapshot['idle_cores']} vs need {need_cores + reserve_cores}, "
                f"avail_mem={snapshot['avail_mem_gib']} GiB vs limit {snapshot['memory_limit_gib']} GiB; "
                f"holders: {holder_desc})"
            )
            return LeaseAcquisitionResult(
                acquired=False,
                lease=None,
                active_holders=active_leases,
                waitlist_position=wait_pos,
                wait_id=wait_id,
                capacity_snapshot=snapshot,
                advisories=adv_list,
                reason=reason,
            )

    def release_eval_lease(self, lease_id: str) -> bool:
        now_iso = utc_now_iso()
        with self.store.immediate_tx() as conn:
            cur = conn.execute(
                "UPDATE eval_leases SET status = 'RELEASED', released_at = ? WHERE lease_id = ? AND status = 'ACTIVE'",
                (now_iso, lease_id),
            )
            changed = (cur.rowcount or 0) > 0
            if changed:
                self._notify_next_waiter_tx(conn, now_iso)
            return changed

    def get_capacity_telemetry(self) -> Dict[str, Any]:
        now_iso = utc_now_iso()
        supply = self._probe_host_supply()
        with self.store.immediate_tx() as conn:
            self._reclaim_stale_leases_tx(conn, now_iso)
            active_rows = conn.execute(
                "SELECT * FROM eval_leases WHERE status = 'ACTIVE' ORDER BY acquired_epoch ASC"
            ).fetchall()
            wait_rows = conn.execute(
                "SELECT * FROM eval_waitlist WHERE status IN ('WAITING', 'NOTIFIED') ORDER BY requested_at ASC"
            ).fetchall()

        leases: List[Dict[str, Any]] = []
        for r in active_rows:
            l_obj = self._row_to_lease(r)
            d = l_obj.model_dump()
            d["fold_progress"] = self.read_ab_partial_progress(l_obj.checkpoint_dir)
            leases.append(d)

        return {
            "supply": supply,
            "active_leases": leases,
            "waitlist": [dict(w) for w in wait_rows],
        }
