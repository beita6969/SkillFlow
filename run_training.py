"""
SkillFlow 训练入口。

用法：
  # 完整训练
  python run_training.py --config configs/skillflow.yaml

  # 仅数据准备
  python run_training.py --prepare-data-only --config configs/skillflow.yaml

  # 从 checkpoint 恢复
  python run_training.py --config configs/skillflow.yaml --resume outputs/skillflow_general/checkpoint_step_0050

  # 快速冒烟测试（3 步）
  python run_training.py --config configs/skillflow.yaml --max-steps 3

  # 仅测试 M_exec 连通
  python run_training.py --config configs/skillflow.yaml --test-connectivity
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

# ── localhost 请求绕过代理（SGLang/M_exec 在 127.0.0.1）──
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import yaml

# ALFWorld 环境变量（必须在 import alfworld 之前设置）
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

_conda_prefix = os.environ.get("CONDA_PREFIX")
if _conda_prefix:
    java_home = Path(_conda_prefix)
    jvm_path = java_home / "lib" / "jvm" / "lib" / "server" / "libjvm.so"
    if java_home.exists():
        os.environ.setdefault("JAVA_HOME", str(java_home))
    if jvm_path.exists():
        os.environ.setdefault("JVM_PATH", str(jvm_path))


def _expand_config_value(value):
    """Expand ${VAR:-default} and standard ${VAR} placeholders in YAML configs."""
    if isinstance(value, dict):
        return {k: _expand_config_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_config_value(v) for v in value]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match) -> str:
        var, default = match.group(1), match.group(2)
        return os.environ.get(var, default or "")

    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}", repl, value)
    return os.path.expandvars(os.path.expanduser(value))


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str) -> dict:
    """加载 YAML 配置，返回展平的 dict"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = _expand_config_value(yaml.safe_load(f))

    # 展平 training 节点
    if "training" in raw:
        return raw["training"]
    return raw


def load_data(path: str) -> list:
    """加载 JSON 数据文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_connectivity(config: dict) -> None:
    """测试 M_exec 和 Supervisor vLLM 连通性"""
    from src.executor.m_exec import MExec

    print("Testing M_exec connectivity...")
    m_exec = MExec(
        api_base=config["executor_api_base"],
        model_name=config.get("executor_model", "gpt-oss-120b"),
    )
    ok = m_exec.test_connectivity()
    print(f"  M_exec: {'OK' if ok else 'FAILED'}")

    print("Testing Supervisor vLLM connectivity...")
    from openai import OpenAI
    try:
        client = OpenAI(
            base_url=config["supervisor_api_base"],
            api_key=config.get("supervisor_api_key") or os.environ.get("SGLANG_API_KEY", "EMPTY"),
        )
        resp = client.chat.completions.create(
            model=config.get("supervisor_model", "Qwen3-8B"),
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        print(f"  Supervisor: OK — {resp.choices[0].message.content}")
    except Exception as e:
        print(f"  Supervisor: FAILED — {e}")


def run_genesis_only(config: dict, train_data: list) -> None:
    """仅运行 genesis，生成初始技能集"""
    from src.executor.m_exec import MExec
    from src.skills.workspace import SkillWorkspace
    from src.skills.skill_creator import SkillCreator

    output_dir = Path(config["output_dir"])
    skills_dir = output_dir / "skills"

    m_exec = MExec(
        api_base=config["executor_api_base"],
        model_name=config.get("executor_model", "gpt-oss-120b"),
    )
    workspace = SkillWorkspace(skills_dir=skills_dir, max_skills=config.get("max_skills_total", 60))
    creator = SkillCreator(m_exec=m_exec, skill_workspace=workspace)

    # 种子样本
    by_type: dict = {}
    for q in train_data:
        tt = q.get("task_type", "unknown")
        if tt not in by_type:
            by_type[tt] = []
        if len(by_type[tt]) < 8:
            by_type[tt].append(q)
    seeds = [q for qs in by_type.values() for q in qs]

    skills = creator.genesis(seeds, target_count=config.get("genesis_count", 12))
    added = workspace.add_batch(skills)
    print(f"Genesis complete: {added} skills added to {skills_dir}")
    for s in skills:
        print(f"  [{s.meta.skill_id}] {s.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillFlow GFlowNet Training")
    parser.add_argument("--config", type=str, default="configs/skillflow.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--fresh", action="store_true",
                        help="从零开始训练：清空旧 skills、日志、checkpoint，全新 genesis + 全新权重")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps (for testing)")
    parser.add_argument("--prepare-data-only", action="store_true", help="Run data preparation only")
    parser.add_argument("--genesis-only", action="store_true", help="Run genesis only")
    parser.add_argument("--test-connectivity", action="store_true", help="Test API connectivity")
    parser.add_argument("--gpu", type=str, default=None, help="Override CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    # GPU 设置
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU(s): {args.gpu}")

    # 配置加载
    config = load_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    setup_logging(config.get("log_level", "INFO"))
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {config.get('exp_name', 'skillflow')}, max_steps={config.get('max_steps')}")

    # 连通性测试
    if args.test_connectivity:
        test_connectivity(config)
        return

    # 数据准备
    if args.prepare_data_only:
        from data.prepare_data import prepare_all
        prepare_all(
            output_dir=Path("data"),
            n_per_task=config.get("n_per_task", 1000),
        )
        return

    # 加载数据
    train_path = config.get("train_data", "data/train.json")
    val_path = config.get("val_data", "data/val.json")

    if not Path(train_path).exists():
        logger.warning(f"Train data not found: {train_path}. Run --prepare-data-only first.")
        logger.info("Creating minimal test data for smoke test...")
        train_data = _create_test_data()
        val_data = _create_test_data(n=10)
    else:
        train_data = load_data(train_path)
        val_data = load_data(val_path) if Path(val_path).exists() else []

    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val samples")

    # Genesis only
    if args.genesis_only:
        run_genesis_only(config, train_data)
        return

    # --fresh：清空旧产物，确保从零开始
    if args.fresh:
        import shutil
        import time
        output_dir = Path(config.get("output_dir", "outputs/skillflow_general"))
        skills_dir = output_dir / "skills"
        backup_tag = int(time.time())

        if args.resume:
            logger.error("--fresh 和 --resume 不能同时使用")
            return

        # 备份旧 skills
        if skills_dir.exists() and any(skills_dir.iterdir()):
            backup = output_dir / f"skills_backup_{backup_tag}"
            shutil.copytree(skills_dir, backup)
            shutil.rmtree(skills_dir)
            skills_dir.mkdir()
            logger.info(f"Fresh: 旧 skills 备份到 {backup}，skills 目录已清空")

        # 备份旧日志
        log_file = output_dir / "training_log.jsonl"
        if log_file.exists() and log_file.stat().st_size > 0:
            log_backup = output_dir / f"training_log.jsonl.bak.{backup_tag}"
            shutil.copy2(log_file, log_backup)
            log_file.write_text("")
            logger.info(f"Fresh: 旧日志备份到 {log_backup}，日志已清空")

        # 备份旧 checkpoints
        for ckpt in output_dir.glob("checkpoint_*"):
            ckpt_backup = output_dir / f"{ckpt.name}_backup_{backup_tag}"
            shutil.copytree(ckpt, ckpt_backup)
            shutil.rmtree(ckpt)
            logger.info(f"Fresh: {ckpt.name} 备份到 {ckpt_backup.name}")

        # 备份旧 trajectory dumps
        dump_dir = output_dir / "trajectory_dumps"
        if dump_dir.exists() and any(dump_dir.iterdir()):
            dump_backup = output_dir / f"trajectory_dumps_backup_{backup_tag}"
            shutil.copytree(dump_dir, dump_backup)
            shutil.rmtree(dump_dir)
            logger.info(f"Fresh: trajectory_dumps 备份到 {dump_backup.name}")

        logger.info("Fresh: 全部旧产物已清空，将从零开始训练（genesis + 全新权重）")

    # 完整训练
    from training.gflownet_trainer import GFlowNetTrainer

    trainer = GFlowNetTrainer(config=config)
    trainer.setup(train_data=train_data, val_data=val_data)

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()


def _create_test_data(n: int = 32) -> list:
    """创建最小测试数据（无 HuggingFace 依赖）"""
    samples = []
    task_types = [
        ("multi_hop_qa", "What is the capital of the country that borders France to the north?", "Belgium"),
        ("fact_checking", "Claim: The Earth orbits the Sun.\nIs this claim SUPPORTS, REFUTES, or NOT ENOUGH INFO?", "supports"),
        ("math_reasoning", "If John has 5 apples and gives away 2, how many does he have?", "3"),
        ("strategy_qa", "Can a person born in 1990 legally vote in the US today?", "yes"),
    ]
    for i in range(n):
        q, question, answer = task_types[i % len(task_types)]
        samples.append({
            "question": question,
            "answer": answer,
            "task_type": q,
            "context": [],
            "extra": {"metric": "token_f1", "eval_fn": "token_f1"},
        })
    return samples


if __name__ == "__main__":
    main()
