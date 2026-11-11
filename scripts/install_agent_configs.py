"""Generates (and optionally installs) identity-bound MCP configs, Git hooks, and agent-bus skills for all 5 checkouts.

Defaults strictly to --dry-run so zero files outside research_loop/ are touched unless --apply is explicitly passed.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.bus.schema import WORKSPACE_AGENTS, AgentId


def build_mcp_json_payload(agent: AgentId, workspace_path: Path, root_dir: Path) -> Dict[str, Any]:
    python_bin = root_dir / "venv" / "bin" / "python"
    server_mod = root_dir / "src" / "bus" / "mcp_server.py"
    config_file = root_dir / "config.yaml"
    return {
        "mcpServers": {
            "research-loop-bus": {
                "command": str(python_bin),
                "args": [
                    str(server_mod),
                    "--agent",
                    agent.value,
                    "--workspace",
                    str(workspace_path),
                    "--config",
                    str(config_file),
                ],
            }
        }
    }


def build_codex_toml_snippet(agent: AgentId, workspace_path: Path, root_dir: Path) -> str:
    python_bin = root_dir / "venv" / "bin" / "python"
    server_mod = root_dir / "src" / "bus" / "mcp_server.py"
    config_file = root_dir / "config.yaml"
    return (
        f"[mcp_servers.research_loop_bus]\n"
        f'command = "{python_bin}"\n'
        f'args = ["{server_mod}", "--agent", "{agent.value}", "--workspace", "{workspace_path}", "--config", "{config_file}"]\n'
    )


def build_main_post_commit_hook(root_dir: Path, main_ws: Path) -> str:
    python_bin = root_dir / "venv" / "bin" / "python"
    cli_py = root_dir / "scripts" / "cli.py"
    return f"""#!/usr/bin/env bash
# Auto-broadcast BASELINE_UPDATED to staging, dev, claude, chatgpt when trading-exp (main) advances.
set -u
BRANCH="$(git -C "{main_ws}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"
if [ "$BRANCH" != "main" ]; then
    exit 0
fi
HEAD_SHA="$(git -C "{main_ws}" rev-parse HEAD 2>/dev/null || echo "")"
PREV_SHA="$(git -C "{main_ws}" rev-parse HEAD~1 2>/dev/null || echo "")"
SUBJ="$(git -C "{main_ws}" log -1 --pretty=%s 2>/dev/null || echo "main updated")"
if [ -n "$HEAD_SHA" ]; then
    "{python_bin}" "{cli_py}" bus send \\
        --from main \\
        --to staging,dev,claude,chatgpt \\
        --intent BASELINE_UPDATED \\
        --subject "main advanced: $SUBJ" \\
        --head-sha "$HEAD_SHA" \\
        ${{PREV_SHA:+--base-sha "$PREV_SHA"}} \\
        --no-auto-audit >/dev/null 2>&1 || true
fi
"""


def generate_bundle(
    *,
    root_dir: Path,
    config: Dict[str, Any],
    output_dir: Path,
    apply_to_workspaces: bool = False,
) -> List[Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ws_confs = config.get("workspaces", {})
    skill_src = root_dir / "skills" / "agent-bus" / "SKILL.md"
    actions: List[Dict[str, str]] = []

    for aid in WORKSPACE_AGENTS:
        conf = ws_confs.get(aid.value)
        if not isinstance(conf, dict) or not conf.get("path"):
            continue
        ws_path = Path(conf["path"]).resolve()
        agent_stage_dir = output_dir / aid.value
        agent_stage_dir.mkdir(parents=True, exist_ok=True)

        # 1. .mcp.json
        mcp_payload = build_mcp_json_payload(aid, ws_path, root_dir)
        staged_mcp = agent_stage_dir / ".mcp.json"
        staged_mcp.write_text(json.dumps(mcp_payload, indent=2) + "\n", encoding="utf-8")
        target_mcp = ws_path / ".mcp.json"
        actions.append(
            {
                "agent": aid.value,
                "kind": "mcp_json",
                "staged": str(staged_mcp),
                "target": str(target_mcp),
                "applied": "no",
            }
        )

        # 2. Codex config snippet for chatgpt
        if aid == AgentId.CHATGPT:
            staged_toml = agent_stage_dir / "codex_mcp_config.toml"
            staged_toml.write_text(build_codex_toml_snippet(aid, ws_path, root_dir), encoding="utf-8")
            actions.append(
                {
                    "agent": aid.value,
                    "kind": "codex_toml_snippet",
                    "staged": str(staged_toml),
                    "target": str(ws_path / ".codex" / "config.toml"),
                    "applied": "no",
                }
            )

        # 3. Shared agent-bus skill
        if skill_src.exists():
            staged_skill = agent_stage_dir / "SKILL.md"
            shutil.copyfile(skill_src, staged_skill)
            target_skill = ws_path / ".agents" / "skills" / "agent-bus" / "SKILL.md"
            actions.append(
                {
                    "agent": aid.value,
                    "kind": "skill",
                    "staged": str(staged_skill),
                    "target": str(target_skill),
                    "applied": "no",
                }
            )

        # 4. post-commit / post-merge hook for main (trading-exp)
        if aid == AgentId.MAIN:
            hook_text = build_main_post_commit_hook(root_dir, ws_path)
            staged_hook = agent_stage_dir / "post-commit"
            staged_hook.write_text(hook_text, encoding="utf-8")
            os.chmod(staged_hook, os.stat(staged_hook).st_mode | stat.S_IXUSR)
            for hook_name in ("post-commit", "post-merge"):
                actions.append(
                    {
                        "agent": aid.value,
                        "kind": f"git_hook_{hook_name}",
                        "staged": str(staged_hook),
                        "target": str(ws_path / ".git" / "hooks" / hook_name),
                        "applied": "no",
                    }
                )

        if apply_to_workspaces and ws_path.exists():
            # Safely merge .mcp.json if one already exists
            if target_mcp.exists():
                try:
                    existing = json.loads(target_mcp.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                existing.setdefault("mcpServers", {}).update(mcp_payload["mcpServers"])
                target_mcp.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
            else:
                shutil.copyfile(staged_mcp, target_mcp)
            actions[-1]["applied"] = "yes"

            if skill_src.exists():
                target_skill = ws_path / ".agents" / "skills" / "agent-bus" / "SKILL.md"
                target_skill.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(skill_src, target_skill)

            if aid == AgentId.MAIN and (ws_path / ".git" / "hooks").exists():
                for hook_name in ("post-commit", "post-merge"):
                    dst_hook = ws_path / ".git" / "hooks" / hook_name
                    if not dst_hook.exists():
                        shutil.copyfile(agent_stage_dir / "post-commit", dst_hook)
                        os.chmod(dst_hook, os.stat(dst_hook).st_mode | stat.S_IXUSR)

            for item in actions:
                if item["agent"] == aid.value and item["kind"] in ("mcp_json", "skill", "git_hook_post-commit", "git_hook_post-merge"):
                    item["applied"] = "yes"

    return actions


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate or install 5-agent MCP & hook configs.")
    ap.add_argument("--config", default=str(ROOT_DIR / "config.yaml"))
    ap.add_argument("--output-dir", default=str(ROOT_DIR / "state" / "generated_agent_configs"))
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write generated configs into the 5 trading* workspaces (default: dry-run staging only).",
    )
    ap.add_argument("--dry-run", action="store_true", default=True, help="Stage configs locally without touching trading* repos.")
    args = ap.parse_args()

    conf = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    do_apply = bool(args.apply)
    actions = generate_bundle(
        root_dir=ROOT_DIR,
        config=conf,
        output_dir=Path(args.output_dir),
        apply_to_workspaces=do_apply,
    )
    mode_lbl = "APPLIED" if do_apply else "DRY-RUN (staged under state/generated_agent_configs/)"
    print(f"\n=== Agent Bus Config Installer [{mode_lbl}] ===")
    for act in actions:
        print(f"  [{act['agent']:<8}] {act['kind']:<20} staged={act['staged']} -> target={act['target']}")
    print()


if __name__ == "__main__":
    main()
