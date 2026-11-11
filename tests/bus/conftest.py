"""Shared 5-checkout Git + BusService test fixtures."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Any, Dict
import pytest
import yaml


def git_run(cwd: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(cwd), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return res.stdout.strip()


@pytest.fixture
def five_repo_topology(tmp_path: Path) -> Dict[str, Any]:
    """
    Builds an isolated 5-checkout Git topology mirroring production:
    - main, staging, dev, claude, chatgpt cloned from a local bare origin.
    - main advances 3 commits AFTER chatgpt is cloned (so chatgpt is 3 behind main).
    - dev creates 1 committed feature change ahead of main.
    """
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, stdout=subprocess.DEVNULL)

    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-b", "main", str(seed)], check=True, stdout=subprocess.DEVNULL)
    git_run(seed, "config", "user.email", "bot@research-loop.local")
    git_run(seed, "config", "user.name", "Research Loop Bot")

    (seed / "models").mkdir(parents=True)
    (seed / "data_access").mkdir(parents=True)
    (seed / "research_notes").mkdir(parents=True)
    (seed / "reports").mkdir(parents=True)
    (seed / "research_notes" / ".gitkeep").write_text("keep\n", encoding="utf-8")
    (seed / "reports" / ".gitkeep").write_text("keep\n", encoding="utf-8")
    (seed / "models" / "ranker.py").write_text("def score(x):\n    return x\n", encoding="utf-8")
    (seed / "data_access" / "features.py").write_text("FEATURES = ['ret_1d']\n", encoding="utf-8")
    git_run(seed, "add", ".")
    git_run(seed, "commit", "-m", "Initial baseline c0")
    c0_sha = git_run(seed, "rev-parse", "HEAD")
    git_run(seed, "push", "-u", str(origin), "main")

    # Clone chatgpt at c0 so it will become 3 commits behind main
    ws_chatgpt = tmp_path / "ws" / "trading-gpt"
    subprocess.run(
        ["git", "clone", "-b", "main", str(origin), str(ws_chatgpt)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    git_run(ws_chatgpt, "checkout", "-b", "chatgpt")
    git_run(ws_chatgpt, "config", "user.email", "chatgpt@research-loop.local")
    git_run(ws_chatgpt, "config", "user.name", "ChatGPT Agent")

    # Advance main by 3 commits in trading-exp
    ws_main = tmp_path / "ws" / "trading-exp"
    subprocess.run(
        ["git", "clone", "-b", "main", str(origin), str(ws_main)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    git_run(ws_main, "config", "user.email", "main@research-loop.local")
    git_run(ws_main, "config", "user.name", "Main Agent")
    for idx in range(1, 4):
        (ws_main / "data_access" / "features.py").write_text(
            f"FEATURES = ['ret_1d', 'lag_{idx}']\n", encoding="utf-8"
        )
        git_run(ws_main, "add", "data_access/features.py")
        git_run(ws_main, "commit", "-m", f"Main advance {idx}")
    main_head = git_run(ws_main, "rev-parse", "HEAD")
    git_run(ws_main, "push", str(origin), "main")

    # Clone staging, dev, claude at latest main_head
    checkouts: Dict[str, Path] = {"main": ws_main, "chatgpt": ws_chatgpt}
    for agent_name, dir_name in [
        ("staging", "trading-staging"),
        ("dev", "trading"),
        ("claude", "trading-claude"),
    ]:
        ws_p = tmp_path / "ws" / dir_name
        subprocess.run(
            ["git", "clone", "-b", "main", str(origin), str(ws_p)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        git_run(ws_p, "checkout", "-b", agent_name)
        git_run(ws_p, "config", "user.email", f"{agent_name}@research-loop.local")
        git_run(ws_p, "config", "user.name", f"{agent_name.title()} Agent")
        checkouts[agent_name] = ws_p

    # Create a committed candidate feature on dev
    ws_dev = checkouts["dev"]
    (ws_dev / "data_access" / "features.py").write_text(
        "FEATURES = ['ret_1d', 'lag_1', 'lag_2', 'lag_3', 'parkinson_vol_lag1']\n",
        encoding="utf-8",
    )
    git_run(ws_dev, "add", "data_access/features.py")
    git_run(ws_dev, "commit", "-m", "Add lag-1 Parkinson volatility feature")
    dev_feature_sha = git_run(ws_dev, "rev-parse", "HEAD")

    hub_root = tmp_path / "hub"
    results_dir = hub_root / "evaluation" / "outputs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    config_dict: Dict[str, Any] = {
        "workspaces": {
            aid: {
                "name": aid.title(),
                "path": str(checkouts[aid]),
                "branch": aid,
                "role": f"{aid} role",
            }
            for aid in ("main", "staging", "dev", "claude", "chatgpt")
        },
        "golden_truth": {"workspace": "main", "branch": "main"},
        "bus": {
            "db_path": str(hub_root / "state" / "hub.db"),
            "threads_dir": str(hub_root / "threads"),
            "knowledge_dir": str(hub_root / "knowledge"),
            "peer_refs_dir": str(hub_root / "state" / "peer_refs.git"),
            "worktree_root": str(tmp_path / "worktrees"),
            "audit_timeout_seconds": 30,
        },
        "evaluation": {
            "results_dir": str(results_dir),
        },
    }
    cfg_path = hub_root / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")

    return {
        "tmp_path": tmp_path,
        "hub_root": hub_root,
        "config_path": cfg_path,
        "config": config_dict,
        "checkouts": checkouts,
        "c0_sha": c0_sha,
        "main_head": main_head,
        "dev_feature_sha": dev_feature_sha,
        "results_dir": results_dir,
    }
