"""
Curated Skills — 手工编写的基线技能。

目录结构：
  configs/curated_skills/
  ├── math-reasoning/SKILL.md
  ├── multi-hop-qa/SKILL.md
  ├── factual-qa/SKILL.md
  ├── science-qa/SKILL.md
  ├── code-generation/SKILL.md
  └── interactive-agent/SKILL.md

每个 SKILL.md 的格式：
  ---
  name: skill-name
  description: "< 250 chars, 用于检索匹配"
  task-type: task_type_name
  source: curated
  ---
  [Markdown body — 实际指令，< 60 words]
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional

_CURATED_CACHE: Optional[Dict[str, str]] = None
_CURATED_DIR = Path(__file__).parent


def load_curated_skills() -> Dict[str, str]:
    """加载所有 curated skills，返回 {task_type: body_text}。

    延迟加载 + 缓存。body_text 是 SKILL.md 的 markdown 正文部分。
    """
    global _CURATED_CACHE
    if _CURATED_CACHE is not None:
        return _CURATED_CACHE

    _CURATED_CACHE = {}
    for skill_dir in _CURATED_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        try:
            text = skill_file.read_text(encoding="utf-8")
            # 解析 frontmatter + body
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1])
                body = parts[2].strip()
                task_type = frontmatter.get("task-type", "")
                if task_type and body:
                    _CURATED_CACHE[task_type] = body
        except Exception:
            pass

    return _CURATED_CACHE


def get_curated_tip(task_type: str) -> str:
    """获取指定 task_type 的 curated tip body。"""
    skills = load_curated_skills()
    return skills.get(task_type, "")
