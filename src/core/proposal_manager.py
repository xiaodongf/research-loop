"""Proposal file manager: loads, serializes, parses, and persists RFC proposals."""
from __future__ import annotations
import os, re, yaml
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

from src.core.schema import (
    Proposal, ProposalMetadata, ProposalStatus, ModelName, StrategyTarget, AcceptanceCriteria
)


class SchemaValidationError(Exception):
    """Raised when an RFC file or LLM response fails validation."""
    pass


class ProposalManager:
    def __init__(self, root_dir: Optional[Path] = None, base_dir: Optional[Path] = None):
        self.root_dir = root_dir or base_dir or Path.cwd()
        if base_dir and not (base_dir / "proposals").exists() and base_dir.name == "proposals":
            # If base_dir already points directly to proposals directory
            self.active_dir = base_dir / "active"
            self.promoted_dir = base_dir / "promoted"
            self.rejected_dir = base_dir / "rejected"
        else:
            self.active_dir = self.root_dir / "proposals" / "active"
            self.promoted_dir = self.root_dir / "proposals" / "promoted"
            self.rejected_dir = self.root_dir / "proposals" / "rejected"

        for d in (self.active_dir, self.promoted_dir, self.rejected_dir):
            d.mkdir(parents=True, exist_ok=True)

    def get_next_proposal_id(self) -> str:
        all_files = list(self.active_dir.glob("PROP-*.md")) + \
                    list(self.promoted_dir.glob("PROP-*.md")) + \
                    list(self.rejected_dir.glob("PROP-*.md"))
        max_num = 0
        for f in all_files:
            match = re.search(r"PROP-(\d+)", f.name)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        return f"PROP-{(max_num + 1):03d}"

    def create_proposal(
        self,
        title: str,
        author_model: ModelName,
        target_strategy: StrategyTarget,
        assigned_branch: str,
        hypothesis: str,
        proposed_files: Optional[List[str]] = None,
        implementation_spec: Optional[str] = None,
        status: ProposalStatus = ProposalStatus.DRAFT,
        acceptance_criteria: Optional[AcceptanceCriteria] = None,
    ) -> Proposal:
        pid = self.get_next_proposal_id()
        meta = ProposalMetadata(
            id=pid,
            title=title,
            author_model=author_model,
            assigned_branch=assigned_branch,
            status=status,
            target_strategy=target_strategy,
            acceptance_criteria=acceptance_criteria or AcceptanceCriteria(),
            proposed_files=proposed_files or [],
        )
        proposal = Proposal(
            metadata=meta,
            hypothesis=hypothesis,
            proposed_files=proposed_files or [],
            implementation_spec=implementation_spec or "",
        )
        self.save_proposal(proposal)
        return proposal

    def serialize_to_markdown(self, proposal: Proposal) -> str:
        meta_dict = proposal.metadata.model_dump(mode="json")
        
        # Serialize frontmatter cleanly as primitive YAML
        yaml_str = yaml.dump(meta_dict, sort_keys=False, default_flow_style=False).strip()
        
        content = f"---\n{yaml_str}\n---\n\n"
        content += f"## Hypothesis\n{proposal.hypothesis.strip()}\n\n"
        
        if proposal.proposed_files:
            content += "## Proposed Code Modifications\n"
            for f in proposal.proposed_files:
                content += f"- `{f}`\n"
            content += "\n"
            
        if proposal.implementation_spec:
            content += f"## Implementation Spec\n{proposal.implementation_spec.strip()}\n\n"
            
        if proposal.discussion_log:
            content += f"## Discussion & Revision Log\n{proposal.discussion_log.strip()}\n\n"
            
        if proposal.scorecard_summary:
            content += "## Scorecard Summary\n```json\n"
            import json
            if hasattr(proposal.scorecard_summary, "model_dump"):
                summary_data = proposal.scorecard_summary.model_dump(mode="json")
            else:
                summary_data = proposal.scorecard_summary
            content += json.dumps(summary_data, indent=2)
            content += "\n```\n"
            
        return content

    def parse_markdown(self, content: str) -> Proposal:
        parts = content.split("---")
        if len(parts) < 3:
            raise SchemaValidationError("Invalid RFC format: missing frontmatter block delimiters ('---').")
        
        yaml_text = parts[1].strip()
        body = "---".join(parts[2:]).strip()

        try:
            meta_data = yaml.safe_load(yaml_text)
        except Exception as e:
            raise SchemaValidationError(f"Invalid YAML frontmatter: {e}")

        if not isinstance(meta_data, dict):
            raise SchemaValidationError("YAML frontmatter must be a key-value dictionary.")

        try:
            metadata = ProposalMetadata(**meta_data)
        except Exception as e:
            raise SchemaValidationError(f"Metadata validation error: {e}")

        # Extract sections from markdown body
        hypothesis = ""
        hypo_match = re.search(r"## Hypothesis\s+(.*?)(?=\n## |\Z)", body, re.DOTALL)
        if hypo_match:
            hypothesis = hypo_match.group(1).strip()

        proposed_files = []
        files_match = re.search(r"## Proposed Code Modifications\s+(.*?)(?=\n## |\Z)", body, re.DOTALL)
        if files_match:
            lines = files_match.group(1).strip().splitlines()
            for line in lines:
                file_m = re.search(r"[-*]\s*`?([a-zA-Z0-9_\-\.\/]+)`?", line)
                if file_m:
                    proposed_files.append(file_m.group(1).strip())

        impl_spec = ""
        impl_match = re.search(r"## Implementation Spec\s+(.*?)(?=\n## |\Z)", body, re.DOTALL)
        if impl_match:
            impl_spec = impl_match.group(1).strip()

        discussion_log = ""
        disc_match = re.search(r"## Discussion & Revision Log\s+(.*?)(?=\n## |\Z)", body, re.DOTALL)
        if disc_match:
            discussion_log = disc_match.group(1).strip()

        scorecard = None
        score_match = re.search(r"## Scorecard Summary\s+```json\s+(.*?)\s+```", body, re.DOTALL)
        if score_match:
            import json
            try:
                scorecard = json.loads(score_match.group(1))
            except Exception:
                pass

        return Proposal(
            metadata=metadata,
            hypothesis=hypothesis,
            proposed_files=proposed_files,
            implementation_spec=impl_spec,
            discussion_log=discussion_log,
            scorecard_summary=scorecard
        )

    def save_proposal(self, proposal: Proposal) -> Path:
        target_dir = self.active_dir
        if proposal.status == ProposalStatus.PROMOTED:
            target_dir = self.promoted_dir
        elif proposal.status == ProposalStatus.REJECTED:
            target_dir = self.rejected_dir

        file_path = target_dir / f"{proposal.id}.md"
        # If moving across directories, clean up old file
        for old_dir in (self.active_dir, self.promoted_dir, self.rejected_dir):
            if old_dir != target_dir:
                old_file = old_dir / f"{proposal.id}.md"
                if old_file.exists():
                    old_file.unlink()

        content = self.serialize_to_markdown(proposal)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def promote_proposal(self, proposal: Proposal) -> Path:
        from src.core.state_machine import ProposalStateMachine
        ProposalStateMachine.promote(proposal)
        return self.save_proposal(proposal)

    def reject_proposal(self, proposal: Proposal, reason: str) -> Path:
        from src.core.state_machine import ProposalStateMachine
        ProposalStateMachine.reject(proposal, reason=reason)
        return self.save_proposal(proposal)

    def load_proposal(self, proposal_id: str) -> Optional[Proposal]:
        for d in (self.active_dir, self.promoted_dir, self.rejected_dir):
            f = d / f"{proposal_id}.md"
            if f.exists():
                return self.parse_markdown(f.read_text(encoding="utf-8"))
        return None

    def list_proposals(self, status: Optional[ProposalStatus] = None) -> List[Proposal]:
        results = []
        for d in (self.active_dir, self.promoted_dir, self.rejected_dir):
            for f in sorted(d.glob("PROP-*.md")):
                try:
                    p = self.parse_markdown(f.read_text(encoding="utf-8"))
                    if status is None or p.status == status:
                        results.append(p)
                except Exception:
                    continue
        return results

    def parse_raw_llm_response(self, raw_text: str, default_author: ModelName, next_id: Optional[str] = None) -> Proposal:
        pid = next_id or self.get_next_proposal_id()
        # If response already contains --- block, parse it directly
        if "---" in raw_text:
            p = self.parse_markdown(raw_text)
            if not p.metadata.id or not p.metadata.id.startswith("PROP-"):
                p.metadata.id = pid
            return p
        
        raise SchemaValidationError("LLM response did not contain required '---' YAML frontmatter block.")
