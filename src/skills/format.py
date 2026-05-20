

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class SkillMeta:


    skill_id: str                          
    source: str                            
    flow_score: float = 0.0               
    reward_variance: float = 0.0          
    creation_step: int = 0
    trajectory_support: List[str] = field(default_factory=list)  
    usage_count: int = 0
    last_used_step: int = -1
    success_count: int = 0               
    task_types: List[str] = field(default_factory=list)  


@dataclass
class TaskTypeSkillDocument:

    task_type: str
    consolidated_strategy: str  
    variant_skill_ids: List[str] = field(default_factory=list)  
    proven_patterns: List[str] = field(default_factory=list)  
    accuracy: float = 0.5
    sample_count: int = 0

    def format_for_prompt(self) -> str:

        lines = [f"### Strategy for {self.task_type} (accuracy: {self.accuracy:.0%}, n={self.sample_count})"]
        if self.consolidated_strategy:
            lines.append(self.consolidated_strategy)
        if self.proven_patterns:
            lines.append("Proven patterns:")
            for p in self.proven_patterns[:3]:
                lines.append(f"  - {p}")
        return "\n".join(lines)


@dataclass
class SkillEntry:


    name: str           
    description: str    
    trigger: str        
    plan: str           
    pitfall: str        
    constraint: str     
    meta: SkillMeta


    def to_markdown(self) -> str:

        meta_dict = {
            "skill_id": self.meta.skill_id,
            "source": self.meta.source,
            "flow_score": round(self.meta.flow_score, 4),
            "reward_variance": round(self.meta.reward_variance, 4),
            "creation_step": self.meta.creation_step,
            "trajectory_support": self.meta.trajectory_support,
            "usage_count": self.meta.usage_count,
            "last_used_step": self.meta.last_used_step,
            "success_count": self.meta.success_count,
            "task_types": self.meta.task_types,
        }
        meta_yaml = yaml.dump(meta_dict, default_flow_style=False).strip()
        plan_indented = "\n".join(f"  {line}" for line in self.plan.strip().split("\n"))

        return (
            f"---\n"
            f'name: "{self.name}"\n'
            f'description: "{self.description}"\n'
            f'trigger: "{self.trigger}"\n'
            f"plan: |\n{plan_indented}\n"
            f'pitfall: "{self.pitfall}"\n'
            f'constraint: "{self.constraint}"\n'
            f"meta:\n"
            + "\n".join(f"  {line}" for line in meta_yaml.split("\n"))
            + "\n---"
        )

    @classmethod
    def from_markdown(cls, text: str) -> "SkillEntry":


        text = re.sub(r"^---\s*", "", text.strip())
        text = re.sub(r"\s*---$", "", text.strip())


        yaml_text = text
        markdown_body = ""
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("#") and i > 0:
                yaml_text = "\n".join(lines[:i])
                markdown_body = "\n".join(lines[i:])
                break


        sanitized_lines = []
        for line in yaml_text.split("\n"):
            if line.startswith(("trigger:", "description:")) and line.count('"') > 2:
                key, _, val = line.partition(":")
                val = val.strip()
                if val.startswith('"') and val.endswith('"'):
                    inner = val[1:-1].replace('"', "'")
                    line = f'{key}: "{inner}"'
            sanitized_lines.append(line)
        yaml_text = "\n".join(sanitized_lines)

        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("Invalid skill YAML")

        meta_data = data.get("meta") or {}
        if isinstance(meta_data, str):
            meta_data = yaml.safe_load(meta_data) or {}

        meta = SkillMeta(
            skill_id=str(meta_data.get("skill_id", "unknown")),
            source=str(meta_data.get("source", "unknown")),
            flow_score=float(meta_data.get("flow_score", 0.0)),
            reward_variance=float(meta_data.get("reward_variance", 0.0)),
            creation_step=int(meta_data.get("creation_step", 0)),
            trajectory_support=list(meta_data.get("trajectory_support") or []),
            usage_count=int(meta_data.get("usage_count", 0)),
            last_used_step=int(meta_data.get("last_used_step", -1)),
            success_count=int(meta_data.get("success_count", 0)),
            task_types=list(meta_data.get("task_types") or []),
        )

        plan = data.get("plan", "")
        if isinstance(plan, str):
            plan = plan.strip()
        trigger = str(data.get("trigger", ""))
        pitfall = str(data.get("pitfall", ""))


        if not trigger or not plan:

            sections = {}
            current_section = ""
            current_content = []
            for line in markdown_body.split("\n"):
                if line.startswith("## "):
                    if current_section:
                        sections[current_section] = "\n".join(current_content).strip()
                    current_section = line[3:].strip().lower()
                    current_content = []
                elif current_section:
                    current_content.append(line)
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()

            if not trigger:
                trigger = sections.get("when to use", "") or sections.get("trigger", "")
            if not plan:
                plan = (
                    sections.get("workflow", "")
                    or sections.get("strategy overview", "")
                    or sections.get("plan", "")
                )

                overview = sections.get("strategy overview", "")
                workflow = sections.get("workflow", "")
                if overview and workflow:
                    plan = f"{overview}\n\n{workflow}"
            if not pitfall:
                pitfall = sections.get("watch out for", "") or sections.get("pitfall", "")

        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            trigger=trigger,
            plan=plan,
            pitfall=pitfall,
            constraint=str(data.get("constraint", "")),
            meta=meta,
        )

    @classmethod
    def from_file(cls, path: Path) -> "SkillEntry":

        content = path.read_text(encoding="utf-8")
        return cls.from_markdown(content)

    def save_to_file(self, path: Path) -> None:

        path.write_text(self.to_markdown(), encoding="utf-8")


    def format_for_prompt(self, include_pitfall: bool = True) -> str:

        usage = self.meta.usage_count
        success_rate = self.meta.success_count / max(usage, 1)
        flow = self.meta.flow_score


        if self.meta.skill_id == "general":
            lines = [
                f"[general] {self.name}",
                f"  When: {self.trigger}",
                f"  Plan: {self.plan}" if self.plan else "",
            ]
            return "\n".join(l for l in lines if l)


        if usage < 5:
            confidence = "new"
        elif success_rate >= 0.7:
            confidence = "reliable"
        elif success_rate >= 0.4:
            confidence = "moderate"
        else:
            confidence = "experimental"
        lines = [
            f"[{self.meta.skill_id}] {self.name}",
            f"  When: {self.trigger}",
            f"  Confidence: {success_rate:.0%} success ({self.meta.success_count}/{usage}), "
            f"flow={flow:.2f} [{confidence}]",
            f"  Plan: {self.plan}" if self.plan else "",
        ]
        return "\n".join(l for l in lines if l)

    def format_brief(self) -> str:

        return f"  [{self.meta.skill_id}] {self.name}: {self.trigger}"


    def validate(self) -> tuple[bool, str]:

        if not self.name:
            return False, "name is empty"
        name_words = len(self.name.split())
        if name_words > 7:
            return False, f"name too long: {name_words} words (max 7)"

        if not self.description:
            return False, "description is empty"

        if not self.trigger:
            return False, "trigger is empty"

        if not self.plan:
            return False, "plan is empty"
        plan_words = len(self.plan.split())
        if plan_words < 20:
            return False, f"plan too short: {plan_words} words (min 20)"
        if plan_words > 200:
            return False, f"plan too long: {plan_words} words (max 200)"


        plan_lower = self.plan.lower()
        _PURE_VAGUE = [
            "gather information systematically",
            "collect all relevant data",
        ]
        vague_count = sum(1 for pat in _PURE_VAGUE if pat in plan_lower)
        plan_words = len(self.plan.split())
        if vague_count >= 2 and plan_words < 40:
            return False, "plan is too vague (multiple generic phrases, no specifics)"


        return True, ""


def parse_skill_blocks(text: str) -> List[SkillEntry]:

    parts = re.split(r"^-{3,}\s*$", text, flags=re.MULTILINE)

    skills: List[SkillEntry] = []
    i = 0
    while i < len(parts):
        block = parts[i].strip()
        i += 1
        if not block or "name:" not in block:
            continue


        full_text = block
        while i < len(parts):
            next_part = parts[i].strip()
            if next_part and ("##" in next_part or next_part.startswith("#")):
                full_text += "\n" + next_part
                i += 1
            else:
                break

        try:
            skill = SkillEntry.from_markdown(f"---\n{full_text}\n---")
            if skill.name:
                skills.append(skill)
        except Exception:
            continue

    return skills


def load_skills_from_dir(skills_dir: Path) -> List[SkillEntry]:

    skills = []
    if not skills_dir.exists():
        return skills

    for path in sorted(skills_dir.glob("*.md")):
        try:
            skill = SkillEntry.from_file(path)
            if skill.name:
                skills.append(skill)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load skill {path}: {e}")

    return skills


_VALID_SKILL_ID_RE = re.compile(r"^dyn_\d{3,}$")


def is_valid_skill_id(skill_id: str) -> bool:

    return bool(skill_id and (_VALID_SKILL_ID_RE.match(skill_id) or skill_id == "general"))


GENERAL_SKILL = SkillEntry(
    name="No Specific Strategy",
    description="Proceed without a specific strategy. Use available tools as you see fit.",
    trigger="When no other strategy seems relevant",
    plan=(
        "1. Analyze the problem on your own.\n"
        "2. Choose and use tools as needed.\n"
        "3. Submit your answer with accept."
    ),
    pitfall="No guidance provided.",
    constraint="No specific constraints.",
    meta=SkillMeta(
        skill_id="general",
        source="system",
        flow_score=0.0,
        usage_count=0,
        success_count=0,
        task_types=[],
    ),
)


def assign_skill_id(skills: List[SkillEntry], start: int = 0) -> None:

    idx = start
    for skill in skills:
        if not is_valid_skill_id(skill.meta.skill_id):
            skill.meta.skill_id = f"dyn_{idx:03d}"
            idx += 1
