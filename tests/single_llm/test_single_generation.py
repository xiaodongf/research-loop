import pytest
from pathlib import Path
from src.core.schema import Proposal, ModelName, StrategyTarget, ProposalStatus
from src.core.proposal_manager import ProposalManager, SchemaValidationError
from src.core.ideation import ContextBuilder, PromptBuilder


def test_context_packet_assembly(tmp_path):
    # Setup mock results and rejected dirs
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "standalone_tb_ab-test_QUICK_20260906.csv").write_text("fold,pnl\n1,10.0")

    rejected_dir = tmp_path / "rejected"
    rejected_dir.mkdir()
    (rejected_dir / "PROP-099.md").write_text("---\nid: PROP-099\n---")

    context = ContextBuilder.build(
        strategy=StrategyTarget.STANDALONE_TB,
        results_dir=results_dir,
        rejected_dir=rejected_dir
    )

    assert "Standalone Triple-Barrier" in context
    assert "standalone_tb_ab-test_QUICK_20260906.csv" in context
    assert "PROP-099" in context


def test_ideation_prompt_formatting():
    prompt = PromptBuilder.create_ideation_prompt(
        author_model=ModelName.CLAUDE,
        strategy=StrategyTarget.STANDALONE_TB,
        theme="Feature Engineering",
        context="Mock Context",
        next_id="PROP-001"
    )

    assert "PROP-001" in prompt
    assert "author_model: claude" in prompt
    assert "assigned_branch: \"claude\"" in prompt
    assert "target_strategy: standalone_tb" in prompt
    assert "## Hypothesis" in prompt


def test_rfc_yaml_schema_validation(tmp_path):
    pm = ProposalManager(root_dir=tmp_path)
    
    mock_llm_output = """---
id: PROP-001
title: "Volatility-Adjusted Momentum"
author_model: claude
assigned_branch: claude
status: HUMAN_REVIEW
target_strategy: standalone_tb
variant_tag: vol_adj_mom
acceptance_criteria:
  eval_mode: quick
  min_pnl_delta: 5.0
  min_win_rate_delta: 0.0
---

## Hypothesis
Adjusting momentum by 20-day volatility reduces false breakouts in choppy regimes.

## Proposed Code Modifications
- `data_access/features.py`
- `evaluation/configs.py`

## Implementation Spec
Compute vol_adj_mom = momentum_10d / vol_20d.shift(1).
"""
    proposal = pm.parse_raw_llm_response(mock_llm_output, default_author=ModelName.CLAUDE, next_id="PROP-001")
    
    assert proposal.id == "PROP-001"
    assert proposal.metadata.title == "Volatility-Adjusted Momentum"
    assert proposal.metadata.author_model == ModelName.CLAUDE
    assert proposal.metadata.status == ProposalStatus.HUMAN_REVIEW
    assert "volatility reduces false breakouts" in proposal.hypothesis
    assert len(proposal.proposed_files) == 2
    assert "data_access/features.py" in proposal.proposed_files


def test_rfc_file_persistence(tmp_path):
    pm = ProposalManager(root_dir=tmp_path)
    mock_llm_output = """---
id: PROP-001
title: "Parkinson Volatility Scaling"
author_model: gemini
assigned_branch: dev
status: HUMAN_REVIEW
target_strategy: standalone_tb
variant_tag: parkinson_vol
---

## Hypothesis
Parkinson volatility scaling stabilizes ranker weights.
"""
    proposal = pm.parse_raw_llm_response(mock_llm_output, default_author=ModelName.GEMINI, next_id="PROP-001")
    saved_path = pm.save_proposal(proposal)

    assert saved_path.exists()
    assert saved_path.name == "PROP-001.md"
    assert saved_path.parent == pm.active_dir

    loaded = pm.load_proposal("PROP-001")
    assert loaded is not None
    assert loaded.id == "PROP-001"
    assert loaded.metadata.title == "Parkinson Volatility Scaling"


def test_malformed_llm_output_handling(tmp_path):
    pm = ProposalManager(root_dir=tmp_path)
    
    # Missing frontmatter delimiters
    bad_output_1 = "I propose we adjust the momentum feature by volatility."
    with pytest.raises(SchemaValidationError):
        pm.parse_raw_llm_response(bad_output_1, default_author=ModelName.CLAUDE)

    # Corrupted YAML
    bad_output_2 = """---
id: PROP-001
title: [unclosed list
author_model: invalid_model
---

## Hypothesis
Some hypothesis.
"""
    with pytest.raises(SchemaValidationError):
        pm.parse_raw_llm_response(bad_output_2, default_author=ModelName.CLAUDE)


def test_ideation_engine_baseline_metrics_extraction(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sample_csv = results_dir / "standalone_tb_ab-test_QUICK_20260906.csv"
    sample_csv.write_text(
        "fold,regime,spy,base_n,base_win,base_total_return,base_prec1,var_n,var_win,var_total_return,var_prec1\n"
        "fold_1,Bull-Hi,0.0,10,0.60,15.0,32.0,10,0.65,20.0,35.0\n"
        "fold_2,Bear-Hi,0.0,10,0.55,-5.0,28.0,10,0.58,2.0,30.0\n"
    )

    metrics = ContextBuilder.get_recent_baseline_metrics(results_dir, StrategyTarget.STANDALONE_TB)
    assert metrics["csv_file"] == sample_csv.name
    assert metrics["strategy"] == "standalone_tb"
    assert metrics["num_folds"] == 2
    assert metrics["top1_precision"] == 30.0
    assert metrics["total_pnl"] == 10.0
    assert metrics["win_rate"] == 57.5
    assert metrics["worst_fold"] == -5.0


def test_ideation_engine_grounded_generation(tmp_path):
    from src.core.ideation import IdeationEngine

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sample_csv = results_dir / "standalone_tb_ab-test_QUICK_20260906.csv"
    sample_csv.write_text(
        "fold,regime,spy,base_n,base_win,base_total_return,base_prec1,var_n,var_win,var_total_return,var_prec1\n"
        "fold_1,Bull-Hi,0.0,10,0.60,15.0,32.0,10,0.65,20.0,35.0\n"
    )

    rejected_dir = tmp_path / "rejected"
    rejected_dir.mkdir()

    prop = IdeationEngine.generate_proposal(
        author_model=ModelName.GEMINI,
        strategy=StrategyTarget.STANDALONE_TB,
        theme="Volatility scaling & normalization (Parkinson / Garman-Klass)",
        results_dir=results_dir,
        rejected_dir=rejected_dir,
        next_id="PROP-010",
    )

    assert prop.id == "PROP-010"
    assert prop.metadata.author_model == ModelName.GEMINI
    assert prop.metadata.assigned_branch == "dev"
    assert "Garman-Klass" in prop.metadata.title
    assert "32.00%" in prop.hypothesis
    assert "\\sigma_{GK" in prop.hypothesis
    assert "calculate_garman_klass_vol" in prop.implementation_spec
    assert prop.metadata.acceptance_criteria.min_pnl_delta == 2.0

