

import os
import yaml
from pathlib import Path
from typing import Dict, Optional

_CURATED_CACHE: Optional[Dict[str, str]] = None
_CURATED_DIR = Path(__file__).parent


def load_curated_skills() -> Dict[str, str]:

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

    skills = load_curated_skills()
    return skills.get(task_type, "")
