"""Parser for evaluation/outputs/results/*.csv files into 10-metric ScorecardSummary models."""
from __future__ import annotations
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from src.core.schema import FoldMetricRow, ScorecardSummary


class InvalidScorecardCSVError(ValueError):
    """Raised when a CSV is not a valid complete A/B fold scorecard (e.g. baseline-only or trade dump)."""


class CSVParser:
    REQUIRED_AB_COLUMNS = frozenset({
        "fold",
        "base_n",
        "base_win",
        "base_total_return",
        "var_n",
        "var_win",
        "var_total_return",
    })

    KNOWN_STRATEGIES = (
        "standalone_tb",
        "stacked_tb",
        "multi_head_tb",
        "quantile",
        "hybrid",
    )

    @staticmethod
    def _parse_finite_float(row: Dict[str, Any], key: str, default: Optional[float] = 0.0) -> float:
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            if default is None:
                raise InvalidScorecardCSVError(f"Missing mandatory numeric value in column '{key}'")
            return float(default)
        try:
            val = float(raw)
        except (TypeError, ValueError) as exc:
            raise InvalidScorecardCSVError(f"Non-numeric value '{raw}' in column '{key}'") from exc
        if not math.isfinite(val):
            raise InvalidScorecardCSVError(f"Non-finite value '{raw}' in column '{key}'")
        return val

    @classmethod
    def _parse_filename_metadata(cls, fname: str) -> tuple[str, str, str, str, str]:
        strategy = "standalone_tb"
        variant = "variant"
        fold_set = "QUICK"
        timestamp = ""
        git_commit = "nogit"

        m = re.match(
            r"([a-zA-Z0-9_]+)_ab-([a-zA-Z0-9_\-]+)_([A-Z0-9\+]+)_(\d{8}_\d{6})_([a-zA-Z0-9]+)\.csv$",
            fname,
        )
        if m:
            return m.groups()  # type: ignore[return-value]

        if "_ab-" in fname:
            strategy, rest = fname[:-4].split("_ab-", 1)
            variant = rest.split("_")[0]
            for stage in ("QUICK", "DEV", "RECENT", "BENCHMARK", "FULL", "MEDIUM"):
                if f"_{stage}_" in rest or rest.endswith(f"_{stage}"):
                    fold_set = stage
                    break
            return strategy, variant, fold_set, timestamp, git_commit

        stem = fname[:-4] if fname.endswith(".csv") else fname
        for known in cls.KNOWN_STRATEGIES:
            if stem.startswith(known + "_"):
                strategy = known
                rem = stem[len(known) + 1 :]
                parts = rem.split("_")
                if parts:
                    variant = parts[0]
                return strategy, variant, fold_set, timestamp, git_commit

        parts = stem.split("_")
        if len(parts) >= 2:
            strategy = parts[0]
            variant = parts[1]
        return strategy, variant, fold_set, timestamp, git_commit

    @classmethod
    def _aggregate_rows(
        cls,
        metric_rows: List[FoldMetricRow],
        *,
        strategy: str,
        variant: str,
        fold_set: str,
        timestamp: str = "",
        git_commit: str = "nogit",
    ) -> ScorecardSummary:
        if not metric_rows:
            raise InvalidScorecardCSVError("Cannot build ScorecardSummary from zero fold rows.")

        bp = np.array([r.base_total_return for r in metric_rows], dtype=float)
        vp = np.array([r.var_total_return for r in metric_rows], dtype=float)
        dp = vp - bp

        bn = np.array([r.base_n for r in metric_rows], dtype=float)
        vn = np.array([r.var_n for r in metric_rows], dtype=float)
        bw = np.array([r.base_win for r in metric_rows], dtype=float)
        vw = np.array([r.var_win for r in metric_rows], dtype=float)

        btwr = float((bw * bn).sum() / bn.sum()) * 100.0 if bn.sum() else 0.0
        vtwr = float((vw * vn).sum() / vn.sum()) * 100.0 if vn.sum() else 0.0
        dtwr = vtwr - btwr

        j = int(np.abs(dp).argmax())
        bp_ex = float(bp.sum() - bp[j])
        vp_ex = float(vp.sum() - vp[j])
        dp_ex = vp_ex - bp_ex

        bprec = np.array([r.base_prec1 for r in metric_rows], dtype=float)
        vprec = np.array([r.var_prec1 for r in metric_rows], dtype=float)
        bprec_m = float(bprec.mean())
        vprec_m = float(vprec.mean())

        bdd = np.array([r.base_max_dd for r in metric_rows], dtype=float)
        vdd = np.array([r.var_max_dd for r in metric_rows], dtype=float)

        bshp = np.array([r.base_sharpe for r in metric_rows], dtype=float)
        vshp = np.array([r.var_sharpe for r in metric_rows], dtype=float)

        bexp = np.array([r.base_avg_exp for r in metric_rows], dtype=float)
        vexp = np.array([r.var_avg_exp for r in metric_rows], dtype=float)

        regimes = [r.regime for r in metric_rows]
        regime_deltas: Dict[str, float] = {}
        for r_name in sorted(set(regimes)):
            if not r_name:
                continue
            mask = [r == r_name for r in regimes]
            regime_deltas[r_name] = round(float(dp[mask].sum()), 2)

        num_folds = len(metric_rows)
        fold_wins = int((dp > 0).sum())
        fold_win_rate = round((fold_wins / num_folds) * 100.0, 2) if num_folds else 0.0

        worst_fold_base = float(bp.min())
        worst_fold_var = float(vp.min())
        worst_dd_base = float(bdd.min())
        worst_dd_var = float(vdd.min())
        mean_dd_base = float(bdd.mean())
        mean_dd_var = float(vdd.mean())
        mean_shp_base = float(bshp.mean())
        mean_shp_var = float(vshp.mean())
        mean_exp_base = float(bexp.mean())
        mean_exp_var = float(vexp.mean())

        return ScorecardSummary(
            variant=variant,
            strategy=strategy,
            fold_set=fold_set,
            timestamp=timestamp,
            git_commit=git_commit,
            num_folds=num_folds,
            base_pnl_total=round(float(bp.sum()), 2),
            var_pnl_total=round(float(vp.sum()), 2),
            delta_pnl_total=round(float(dp.sum()), 2),
            base_pnl_ex_largest=round(bp_ex, 2),
            var_pnl_ex_largest=round(vp_ex, 2),
            delta_pnl_ex_largest=round(dp_ex, 2),
            base_win_rate=round(btwr, 2),
            var_win_rate=round(vtwr, 2),
            delta_win_rate=round(dtwr, 2),
            fold_wins=fold_wins,
            fold_win_rate=fold_win_rate,
            worst_fold_base=round(worst_fold_base, 2),
            worst_fold_var=round(worst_fold_var, 2),
            delta_worst_fold=round(worst_fold_var - worst_fold_base, 2),
            worst_dd_base=round(worst_dd_base, 2),
            worst_dd_var=round(worst_dd_var, 2),
            delta_worst_dd=round(worst_dd_var - worst_dd_base, 2),
            mean_dd_base=round(mean_dd_base, 2),
            mean_dd_var=round(mean_dd_var, 2),
            delta_mean_dd=round(mean_dd_var - mean_dd_base, 2),
            mean_sharpe_base=round(mean_shp_base, 4),
            mean_sharpe_var=round(mean_shp_var, 4),
            delta_mean_sharpe=round(mean_shp_var - mean_shp_base, 4),
            mean_exp_base=round(mean_exp_base, 2),
            mean_exp_var=round(mean_exp_var, 2),
            delta_mean_exp=round(mean_exp_var - mean_exp_base, 2),
            base_prec1_mean=round(bprec_m, 2),
            var_prec1_mean=round(vprec_m, 2),
            delta_prec1_mean=round(vprec_m - bprec_m, 2),
            total_trades_base=int(bn.sum()),
            total_trades_var=int(vn.sum()),
            regime_deltas=regime_deltas,
            rows=metric_rows,
        )

    @classmethod
    def parse_result_file(cls, csv_path: Path) -> ScorecardSummary:
        if not csv_path.exists():
            raise FileNotFoundError(f"Result CSV file does not exist: {csv_path}")

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            rows: List[Dict[str, Any]] = list(reader)

        if not rows:
            raise InvalidScorecardCSVError(f"Result CSV file is empty: {csv_path}")

        missing_cols = cls.REQUIRED_AB_COLUMNS - fieldnames
        if missing_cols:
            raise InvalidScorecardCSVError(
                f"CSV '{csv_path.name}' is not a valid A/B fold scorecard; "
                f"missing mandatory columns: {sorted(missing_cols)}"
            )

        strategy, variant, fold_set, timestamp, git_commit = cls._parse_filename_metadata(csv_path.name)

        metric_rows: List[FoldMetricRow] = []
        for r in rows:
            fold = str(r.get("fold", "")).strip()
            if not fold:
                raise InvalidScorecardCSVError(f"Row with empty 'fold' identifier in {csv_path.name}")
            regime = str(r.get("regime", "")).strip()
            spy = cls._parse_finite_float(r, "spy", 0.0)

            b_ret = cls._parse_finite_float(r, "base_total_return", None)
            v_ret = cls._parse_finite_float(r, "var_total_return", None)
            b_win = cls._parse_finite_float(r, "base_win", None)
            v_win = cls._parse_finite_float(r, "var_win", None)
            b_n = int(cls._parse_finite_float(r, "base_n", None))
            v_n = int(cls._parse_finite_float(r, "var_n", None))

            b_prec = cls._parse_finite_float(r, "base_prec1", 0.0)
            v_prec = cls._parse_finite_float(r, "var_prec1", 0.0)
            b_dd = cls._parse_finite_float(r, "base_max_dd", 0.0)
            v_dd = cls._parse_finite_float(r, "var_max_dd", 0.0)
            b_shp = cls._parse_finite_float(r, "base_sharpe", 0.0)
            v_shp = cls._parse_finite_float(r, "var_sharpe", 0.0)
            b_exp = cls._parse_finite_float(r, "base_avg_exp", 0.0)
            v_exp = cls._parse_finite_float(r, "var_avg_exp", 0.0)

            metric_rows.append(
                FoldMetricRow(
                    fold=fold,
                    regime=regime,
                    spy=spy,
                    base_n=b_n,
                    base_win=b_win,
                    base_total_return=b_ret,
                    base_max_dd=b_dd,
                    base_sharpe=b_shp,
                    base_prec1=b_prec,
                    base_avg_exp=b_exp,
                    var_n=v_n,
                    var_win=v_win,
                    var_total_return=v_ret,
                    var_max_dd=v_dd,
                    var_sharpe=v_shp,
                    var_prec1=v_prec,
                    var_avg_exp=v_exp,
                    delta_total_return=v_ret - b_ret,
                    delta_win=(v_win - b_win) * 100.0,
                    delta_prec1=v_prec - b_prec,
                )
            )

        return cls._aggregate_rows(
            metric_rows,
            strategy=strategy,
            variant=variant,
            fold_set=fold_set,
            timestamp=timestamp,
            git_commit=git_commit,
        )

    @classmethod
    def parse_staged_run(cls, csv_paths: Iterable[Path]) -> Dict[str, ScorecardSummary]:
        """
        Parses multiple staged CSVs (e.g. DEV, RECENT, BENCHMARK) for a single run and
        computes a combined TOTAL 10-metric scorecard across unique non-overlapping DEV+RECENT folds.
        """
        stage_summaries: Dict[str, ScorecardSummary] = {}
        for p in csv_paths:
            sc = cls.parse_result_file(p)
            stage_summaries[sc.fold_set.upper()] = sc

        if not stage_summaries:
            raise InvalidScorecardCSVError("No stage CSVs provided to parse_staged_run.")

        # Build TOTAL from DEV + RECENT if present, otherwise across unique fold dates
        primary_stages = [s for k, s in stage_summaries.items() if k in ("DEV", "RECENT")]
        if not primary_stages:
            primary_stages = list(stage_summaries.values())

        seen_folds: set[str] = set()
        combined_rows: List[FoldMetricRow] = []
        for s in primary_stages:
            for r in s.rows:
                if r.fold not in seen_folds:
                    seen_folds.add(r.fold)
                    combined_rows.append(r)

        first = primary_stages[0]
        if len(stage_summaries) > 1 or "TOTAL" not in stage_summaries:
            stage_summaries["TOTAL"] = cls._aggregate_rows(
                combined_rows,
                strategy=first.strategy,
                variant=first.variant,
                fold_set="TOTAL",
                timestamp=first.timestamp,
                git_commit=first.git_commit,
            )

        return stage_summaries
