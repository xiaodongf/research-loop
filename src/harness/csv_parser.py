"""Parser for evaluation/outputs/results/*.csv files into ScorecardSummary models."""
from __future__ import annotations
import csv, re
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from src.core.schema import FoldMetricRow, ScorecardSummary


class CSVParser:
    @staticmethod
    def parse_result_file(csv_path: Path) -> ScorecardSummary:
        if not csv_path.exists():
            raise FileNotFoundError(f"Result CSV file does not exist: {csv_path}")

        rows: List[Dict[str, Any]] = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        if not rows:
            raise ValueError(f"Result CSV file is empty: {csv_path}")

        # Parse filename metadata e.g. standalone_tb_ab-regime_gated_option_4_QUICK_20260825_225229_5c981c2.csv
        fname = csv_path.name
        strategy = "standalone_tb"
        variant = "variant"
        fold_set = "QUICK"
        timestamp = ""
        git_commit = "nogit"

        m = re.match(r"([a-zA-Z0-9_]+)_ab-([a-zA-Z0-9_]+)_([A-Z0-9\+]+)_(\d{8}_\d{6})_([a-zA-Z0-9]+)\.csv", fname)
        if m:
            strategy, variant, fold_set, timestamp, git_commit = m.groups()
        else:
            # Fallback simple split
            parts = fname.replace(".csv", "").split("_")
            if len(parts) >= 3:
                strategy = parts[0]
                variant = parts[1]

        metric_rows: List[FoldMetricRow] = []
        bp_list = []
        vp_list = []
        bn_list = []
        vn_list = []
        bwin_list = []
        vwin_list = []
        bprec_list = []
        vprec_list = []
        regimes = []

        for r in rows:
            fold = r.get("fold", "")
            regime = r.get("regime", "")
            spy = float(r.get("spy", 0.0))

            b_ret = float(r.get("base_total_return", 0.0))
            v_ret = float(r.get("var_total_return", 0.0))
            d_ret = v_ret - b_ret

            b_win = float(r.get("base_win", 0.0))
            v_win = float(r.get("var_win", 0.0))
            d_win = (v_win - b_win) * 100.0  # as percentage points

            b_prec = float(r.get("base_prec1", 0.0))
            v_prec = float(r.get("var_prec1", 0.0))
            d_prec = v_prec - b_prec

            b_n = int(r.get("base_n", 0))
            v_n = int(r.get("var_n", 0))

            b_dd = float(r.get("base_max_dd", 0.0))
            v_dd = float(r.get("var_max_dd", 0.0))

            b_shp = float(r.get("base_sharpe", 0.0))
            v_shp = float(r.get("var_sharpe", 0.0))

            bp_list.append(b_ret)
            vp_list.append(v_ret)
            bn_list.append(b_n)
            vn_list.append(v_n)
            bwin_list.append(b_win)
            vwin_list.append(v_win)
            bprec_list.append(b_prec)
            vprec_list.append(v_prec)
            regimes.append(regime)

            metric_rows.append(FoldMetricRow(
                fold=fold,
                regime=regime,
                spy=spy,
                base_n=b_n,
                base_win=b_win,
                base_total_return=b_ret,
                base_max_dd=b_dd,
                base_sharpe=b_shp,
                base_prec1=b_prec,
                var_n=v_n,
                var_win=v_win,
                var_total_return=v_ret,
                var_max_dd=v_dd,
                var_sharpe=v_shp,
                var_prec1=v_prec,
                delta_total_return=d_ret,
                delta_win=d_win,
                delta_prec1=d_prec
            ))

        bp = np.array(bp_list)
        vp = np.array(vp_list)
        dp = vp - bp

        bn = np.array(bn_list)
        vn = np.array(vn_list)
        bw = np.array(bwin_list)
        vw = np.array(vwin_list)

        btwr = float((bw * bn).sum() / bn.sum()) * 100.0 if bn.sum() else 0.0
        vtwr = float((vw * vn).sum() / vn.sum()) * 100.0 if vn.sum() else 0.0
        dtwr = vtwr - btwr

        # Ex-largest move
        j = int(np.abs(dp).argmax()) if len(dp) else 0
        bp_ex = float(bp.sum() - bp[j]) if len(bp) else 0.0
        vp_ex = float(vp.sum() - vp[j]) if len(vp) else 0.0
        dp_ex = vp_ex - bp_ex

        # Prec@1
        bprec = np.array(bprec_list)
        vprec = np.array(vprec_list)
        bprec_m = float(bprec.mean()) if len(bprec) else 0.0
        vprec_m = float(vprec.mean()) if len(vprec) else 0.0
        dprec_m = vprec_m - bprec_m

        # Regime deltas
        regime_deltas: Dict[str, float] = {}
        for r_name in sorted(set(regimes)):
            mask = [r == r_name for r in regimes]
            regime_deltas[r_name] = round(float(dp[mask].sum()), 2)

        fold_wins = int((dp > 0).sum())

        return ScorecardSummary(
            variant=variant,
            strategy=strategy,
            fold_set=fold_set,
            timestamp=timestamp,
            git_commit=git_commit,
            num_folds=len(rows),
            base_pnl_total=round(float(bp.sum()), 2),
            var_pnl_total=round(float(vp.sum()), 2),
            delta_pnl_total=round(float(dp.sum()), 2),
            base_pnl_ex_largest=round(bp_ex, 2),
            var_pnl_ex_largest=round(vp_ex, 2),
            delta_pnl_ex_largest=round(dp_ex, 2),
            base_win_rate=round(btwr, 2),
            var_win_rate=round(vtwr, 2),
            delta_win_rate=round(dtwr, 2),
            base_prec1_mean=round(bprec_m, 2),
            var_prec1_mean=round(vprec_m, 2),
            delta_prec1_mean=round(dprec_m, 2),
            worst_fold_base=round(float(bp.min()), 2),
            worst_fold_var=round(float(vp.min()), 2),
            fold_wins=fold_wins,
            regime_deltas=regime_deltas,
            rows=metric_rows
        )
