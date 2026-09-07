import pytest, sys, os, time
from pathlib import Path
from src.harness.csv_parser import CSVParser
from src.harness.eval_bridge import EvaluationBridge, EvaluationTimeoutError, EvaluationExecutionError


def test_csv_parser_synthetic(tmp_path):
    csv_file = tmp_path / "standalone_tb_ab-test_QUICK_20260906_120000_abc123.csv"
    csv_content = """fold,regime,spy,base_n,base_win,base_total_return,base_ret_per_trade,base_max_dd,base_avg_exp,base_max_exp,base_sharpe,base_prec1,var_n,var_win,var_total_return,var_ret_per_trade,var_max_dd,var_avg_exp,var_max_exp,var_sharpe,var_prec1
2012-07-01,bull-lo,3.4,30,0.60,10.0,3.3,-5.0,50.0,100.0,0.5,30.0,30,0.70,15.0,5.0,-4.0,50.0,100.0,0.6,35.0
2020-09-01,bull-hi,-6.0,40,0.50,5.0,1.25,-6.0,40.0,80.0,0.3,20.0,40,0.60,10.0,2.5,-4.0,40.0,80.0,0.4,25.0
"""
    csv_file.write_text(csv_content)

    scorecard = CSVParser.parse_result_file(csv_file)

    assert scorecard.strategy == "standalone_tb"
    assert scorecard.variant == "test"
    assert scorecard.fold_set == "QUICK"
    assert scorecard.num_folds == 2
    assert scorecard.base_pnl_total == 15.0  # 10 + 5
    assert scorecard.var_pnl_total == 25.0   # 15 + 10
    assert scorecard.delta_pnl_total == 10.0 # 25 - 15
    assert scorecard.fold_wins == 2
    assert "bull-lo" in scorecard.regime_deltas
    assert scorecard.regime_deltas["bull-lo"] == 5.0
    assert scorecard.regime_deltas["bull-hi"] == 5.0


def test_eval_bridge_mock_run(tmp_path):
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create mock python script simulating evaluation.cli
    mock_py = tmp_path / "mock_python.sh"
    mock_py.write_text(f"""#!/bin/bash
cat <<'CSVDATA' > "{results_dir}/standalone_tb_ab-aa_QUICK_20260906_120000_mock1.csv"
fold,regime,spy,base_n,base_win,base_total_return,base_ret_per_trade,base_max_dd,base_avg_exp,base_max_exp,base_sharpe,base_prec1,var_n,var_win,var_total_return,var_ret_per_trade,var_max_dd,var_avg_exp,var_max_exp,var_sharpe,var_prec1
2012-07-01,bull-lo,3.4,30,0.6,10.0,3.3,-5.0,50.0,100.0,0.5,30.0,30,0.6,10.0,3.3,-5.0,50.0,100.0,0.5,30.0
CSVDATA
echo "Mock evaluation complete!"
""")
    mock_py.chmod(0o755)

    bridge = EvaluationBridge(python_bin=mock_py, results_dir=results_dir)
    logs = []
    scorecard = bridge.run_evaluation(
        workspace_path=ws_dir,
        strategy="standalone_tb",
        mode="quick",
        ab_variant="aa",
        stdout_callback=lambda line: logs.append(line)
    )

    assert scorecard.variant == "aa"
    assert scorecard.delta_pnl_total == 0.0
    assert any("Mock evaluation complete!" in line for line in logs)


def test_eval_bridge_timeout(tmp_path):
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create sleep script to test timeout
    sleep_py = tmp_path / "sleep.sh"
    sleep_py.write_text("#!/bin/bash\nsleep 5\n")
    sleep_py.chmod(0o755)

    bridge = EvaluationBridge(python_bin=sleep_py, results_dir=results_dir)
    with pytest.raises(EvaluationTimeoutError):
        bridge.run_evaluation(workspace_path=ws_dir, timeout=1)
