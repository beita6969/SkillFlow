"""SkillFlow 技能格式定义 — 7-tuple SkillEntry。

技能是 SkillFlow 中唯一的能力抽象（S₀=∅，从空集自举）。
每个 skill 指导 Supervisor 如何调用 M_exec 解决特定类型的子任务。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class SkillMeta:
    """技能元数据（生命周期、flow 分数等）"""

    skill_id: str                          # dyn_NNN 格式
    source: str                            # genesis | flow_evolution | backward_mining
    flow_score: float = 0.0               # F̂(s) 当前估计值（EMA）
    reward_variance: float = 0.0          # Var_{τ∈B_s}[R(τ)] — 论文 Eq.11
    creation_step: int = 0
    trajectory_support: List[str] = field(default_factory=list)  # 支撑轨迹 IDs
    usage_count: int = 0
    last_used_step: int = -1
    success_count: int = 0               # invoke 后 episode 成功的次数
    task_types: List[str] = field(default_factory=list)  # 适用的 task_types（自动标记）


@dataclass
class TaskTypeSkillDocument:
    """
    XSkill Living Document — 每个 task_type 的合并策略文档。

    XSkill 核心理念："living document that grows wiser"。
    将同一 task_type 下的多个 skill 合并为一个策略摘要，
    包含从 experience store 提取的 proven patterns。
    """
    task_type: str
    consolidated_strategy: str  # XSkill 风格的合并策略摘要
    variant_skill_ids: List[str] = field(default_factory=list)  # 该 type 下所有 skill IDs
    proven_patterns: List[str] = field(default_factory=list)  # 从 experience store 提取的规则
    accuracy: float = 0.5
    sample_count: int = 0

    def format_for_prompt(self) -> str:
        """格式化为 prompt 注入的背景信息"""
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
    """
    SkillFlow 技能 7-tuple：
    (name, description, trigger, plan, pitfall, constraint, meta)

    plan 字段是核心：指导 Supervisor 如何分步向 M_exec 发出 instruction。
    """

    name: str           # 技能名称（≤7 词）
    description: str    # 一句话描述
    trigger: str        # 触发条件（什么场景使用此技能）
    plan: str           # 分步执行策略
    pitfall: str        # 常见失败模式
    constraint: str     # 硬约束（最大步骤数等）
    meta: SkillMeta

    # ──────────────────────────────────────────────
    # 序列化 / 反序列化
    # ──────────────────────────────────────────────

    def to_markdown(self) -> str:
        """序列化为 SKILL.md 格式（用于持久化存储）"""
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
        """从 SKILL.md 格式解析（YAML header + optional markdown body）"""
        # 去除前后 ---
        text = re.sub(r"^---\s*", "", text.strip())
        text = re.sub(r"\s*---$", "", text.strip())

        # 分离 YAML header 和 markdown body
        # YAML 部分：开头到第一个 # 标题行之前
        # Markdown 部分：# 标题行及之后的内容
        yaml_text = text
        markdown_body = ""
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("#") and i > 0:
                yaml_text = "\n".join(lines[:i])
                markdown_body = "\n".join(lines[i:])
                break

        # Sanitize YAML
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

        # Fallback: GENERATE_SKILL_PROMPT 的输出格式将 trigger/plan/pitfall 放在
        # markdown 正文（"When to Use"/"Workflow"/"Watch Out For"）而非 YAML 块中。
        # 如果 YAML 中缺少这些字段，从原始文本的 markdown 段落提取。
        if not trigger or not plan:
            # 从 markdown body 中提取段落（## When to Use, ## Workflow, ## Watch Out For）
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
                # Combine strategy overview + workflow if both exist
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
        """从单个 .md 文件加载"""
        content = path.read_text(encoding="utf-8")
        return cls.from_markdown(content)

    def save_to_file(self, path: Path) -> None:
        """保存到 .md 文件"""
        path.write_text(self.to_markdown(), encoding="utf-8")

    # ──────────────────────────────────────────────
    # Prompt 格式化
    # ──────────────────────────────────────────────

    def format_for_prompt(self, include_pitfall: bool = True) -> str:
        """注入到 Supervisor context — 展示 success_rate/flow 信息。
        模型必须从可用 skills 中选一个，但自主决定选哪个。"""
        usage = self.meta.usage_count
        success_rate = self.meta.success_count / max(usage, 1)
        flow = self.meta.flow_score

        # v5.2: skill 作为"knowledge"呈现，不再是 invoke 目标
        # general skill 使用特殊展示
        if self.meta.skill_id == "general":
            lines = [
                f"[general] {self.name}",
                f"  When: {self.trigger}",
                f"  Plan: {self.plan}" if self.plan else "",
            ]
            return "\n".join(l for l in lines if l)

        # 置信度标签
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
        """超简洁版本（仅 ID + 名称 + trigger，用于技能列表）"""
        return f"  [{self.meta.skill_id}] {self.name}: {self.trigger}"

    # ──────────────────────────────────────────────
    # 质量检查
    # ──────────────────────────────────────────────

    def validate(self) -> tuple[bool, str]:
        """
        检查技能结构完整性。

        Returns:
            (is_valid, error_message)
        """
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

        # 反模糊检测 — 仅拦截纯空洞的 plan（整体无具体步骤）
        # 注：在 tool-augmented 场景中，"verify"/"validate" 是合理的工具操作，
        # 只有当 plan 整体就是空洞建议时才拒绝
        plan_lower = self.plan.lower()
        _PURE_VAGUE = [
            "gather information systematically",
            "collect all relevant data",
        ]
        vague_count = sum(1 for pat in _PURE_VAGUE if pat in plan_lower)
        plan_words = len(self.plan.split())
        if vague_count >= 2 and plan_words < 40:
            return False, "plan is too vague (multiple generic phrases, no specifics)"

        # 注意: 不在 validate() 中检查 skill_id 格式。
        # skill_id 由系统通过 assign_skill_id() 在质量检查之后分配，
        # validate() 只负责检查 LLM 生成的内容质量（name, plan, trigger 等）。

        return True, ""


# ──────────────────────────────────────────────────────
# 批量解析工具函数
# ──────────────────────────────────────────────────────


def parse_skill_blocks(text: str) -> List[SkillEntry]:
    """从一段文本中提取所有 ---...--- 格式的技能块。

    GENERATE_SKILL_PROMPT 的输出格式：
      ---
      name: ...
      description: ...
      ---
      # Skill Title
      ## When to Use ...
      ## Workflow ...
      ## Watch Out For ...

    YAML block 和 markdown body 被 --- 分隔符分开。需要合并后传给 from_markdown。
    """
    parts = re.split(r"^-{3,}\s*$", text, flags=re.MULTILINE)

    skills: List[SkillEntry] = []
    i = 0
    while i < len(parts):
        block = parts[i].strip()
        i += 1
        if not block or "name:" not in block:
            continue

        # 合并后续的 markdown body（## 开头的段落）
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
    """从目录加载所有 .md 技能文件"""
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
    """检查 skill_id 是否是系统生成的合法格式（dyn_NNN 或 general）"""
    return bool(skill_id and (_VALID_SKILL_ID_RE.match(skill_id) or skill_id == "general"))


# ── 通用兜底 Skill（空白占位符，始终可选）──
# 设计意图：提供合法的 skill_invoke 目标，但不包含有价值的策略。
# GFlowNet 训练中，特化 skill 的 reward 高于 general → flow 自然流向特化 skill。
# 随着进化产生足够多的特化 skill，general 的使用率自然趋近 0。
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
    """为没有合法 skill_id 的技能分配 dyn_NNN ID（原地修改）。

    合法 ID 格式为 dyn_NNN（如 dyn_000, dyn_012）。
    任何不符合此格式的 ID（包括 LLM 生成的 placeholder、unknown 等）
    都会被重新分配，确保每个 skill 拥有唯一的系统 ID。
    """
    idx = start
    for skill in skills:
        if not is_valid_skill_id(skill.meta.skill_id):
            skill.meta.skill_id = f"dyn_{idx:03d}"
            idx += 1
