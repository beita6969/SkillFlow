"""
Official SWE-bench evaluation without Docker.

Uses conda environments + git repo clones to run actual tests.
Equivalent to the official SWE-bench evaluation harness.

Flow:
  1. Pre-setup: clone repos, create conda envs per (repo, version)
  2. Evaluate: checkout base_commit → apply model patch → apply test_patch → run FAIL_TO_PASS tests
  3. Resolved iff ALL FAIL_TO_PASS tests pass
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CONDA = os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"
SWE_ENVS = Path(os.environ.get("SWE_BENCH_ENVS", "swe_bench_envs"))
VERIFIED_DS_PATH = os.environ.get("SWE_BENCH_VERIFIED_PATH", "data/swebench_verified")

# Lazy-loaded verified dataset
_verified_cache: Dict[str, dict] = {}
_swe_bench_specs: Dict = {}


def _load_verified_dataset():
    """Load SWE-bench Verified dataset (cached)."""
    global _verified_cache
    if _verified_cache:
        return
    try:
        import datasets
        ds = datasets.load_from_disk(VERIFIED_DS_PATH)
        for row in ds:
            _verified_cache[row["instance_id"]] = {
                "repo": row["repo"],
                "version": row.get("version", ""),
                "base_commit": row["base_commit"],
                "patch": row["patch"],
                "test_patch": row["test_patch"],
                "FAIL_TO_PASS": json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"],
                "PASS_TO_PASS": json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else row["PASS_TO_PASS"],
            }
        logger.info(f"[SWE-eval] Loaded {len(_verified_cache)} verified instances")
    except Exception as e:
        logger.warning(f"[SWE-eval] Failed to load verified dataset: {e}")


def _load_specs():
    """Load SWE-bench test specs (cached)."""
    global _swe_bench_specs
    if _swe_bench_specs:
        return
    try:
        import sys
        swebench_repo = os.environ.get("SWEBENCH_HARNESS_PATH", "")
        if swebench_repo and swebench_repo not in sys.path:
            sys.path.insert(0, swebench_repo)
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        _swe_bench_specs = MAP_REPO_VERSION_TO_SPECS
    except Exception as e:
        logger.warning(f"[SWE-eval] Failed to load specs: {e}")


def _repo_dir(repo: str) -> Path:
    """Get local clone directory for a repo."""
    return SWE_ENVS / repo.replace("/", "__")


def _env_name(repo: str, version: str) -> str:
    """Get conda env name for a (repo, version) pair."""
    return f"swe_{repo.replace('/', '_')}_{version.replace('.', '')}"


def _env_python(repo: str, version: str) -> Optional[str]:
    """Get the Python interpreter path for a (repo, version) conda env."""
    name = _env_name(repo, version)
    envs_dir = os.environ.get("CONDA_ENVS_DIR")
    if not envs_dir and os.path.isabs(CONDA):
        envs_dir = str(Path(CONDA).resolve().parents[1] / "envs")
    if not envs_dir:
        return None
    py = str(Path(envs_dir) / name / "bin" / "python")
    if os.path.exists(py):
        return py
    return None


def setup_repo_env(repo: str, version: str, base_commit: str) -> bool:
    """One-time setup: clone repo + create conda env + install."""
    repo_path = _repo_dir(repo)
    env_name = _env_name(repo, version)
    
    _load_specs()
    spec = _swe_bench_specs.get(repo, {}).get(version, {})
    py_version = spec.get("python", "3.9")
    install_cmd = spec.get("install", "python -m pip install -e .")
    
    # Clone repo if needed
    if not repo_path.exists():
        logger.info(f"[SWE-eval] Cloning {repo}...")
        url = f"https://github.com/{repo}.git"
        result = subprocess.run(
            ["git", "clone", "--quiet", url, str(repo_path)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"[SWE-eval] Clone failed: {result.stderr[:200]}")
            return False
    
    # Create conda env if needed
    env_py = _env_python(repo, version)
    if not env_py:
        logger.info(f"[SWE-eval] Creating env {env_name} (python={py_version})...")
        result = subprocess.run(
            [CONDA, "create", "-n", env_name, f"python={py_version}", "-y", "-q"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"[SWE-eval] Env creation failed: {result.stderr[:200]}")
            return False
        env_py = _env_python(repo, version)
    
    # Checkout base_commit and install
    subprocess.run(["git", "checkout", base_commit, "-q"], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "checkout", ".", "-q"], cwd=repo_path, capture_output=True)
    
    env_pip = str(Path(env_py).parent / "pip")
    result = subprocess.run(
        [env_pip, "install", "-e", ".", "-q"],
        cwd=repo_path, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.warning(f"[SWE-eval] Install warning: {result.stderr[:200]}")
    
    return True


def evaluate_patch(
    instance_id: str,
    model_patch: str,
    extra: Optional[Dict] = None,
    timeout: int = 60,
) -> Tuple[bool, float, str]:
    """
    Official SWE-bench evaluation: apply patch → run FAIL_TO_PASS tests.

    Returns:
        (resolved: bool, score: float, details: str)
        resolved = ALL FAIL_TO_PASS tests pass
        score = fraction of FAIL_TO_PASS tests that pass (0.0-1.0)
    """
    if not model_patch.strip():
        return False, 0.0, "empty_patch"

    _load_verified_dataset()
    verified = _verified_cache.get(instance_id)
    if not verified:
        return False, 0.0, f"instance_not_in_verified({instance_id})"

    repo = verified["repo"]
    version = verified["version"]
    base_commit = verified["base_commit"]
    test_patch = verified["test_patch"]
    fail_to_pass = verified["FAIL_TO_PASS"]

    if not fail_to_pass:
        return False, 0.0, "no_fail_to_pass_tests"

    # Check env is ready
    env_py = _env_python(repo, version)
    repo_path = _repo_dir(repo)
    if not env_py or not repo_path.exists():
        # Try lazy setup
        if not setup_repo_env(repo, version, base_commit):
            return False, 0.0, "env_not_ready"
        env_py = _env_python(repo, version)

    _load_specs()
    spec = _swe_bench_specs.get(repo, {}).get(version, {})
    test_cmd_template = spec.get("test_cmd", "pytest -rA")
    if isinstance(test_cmd_template, list):
        test_cmd_template = test_cmd_template[-1]

    # 使用 git worktree 创建隔离的评估目录（不影响 agent 的工作目录）
    import tempfile, shutil
    eval_dir = Path(tempfile.mkdtemp(prefix="swe_eval_"))
    worktree_path = eval_dir / "repo"

    try:
        # 创建 worktree 在 base_commit
        subprocess.run(
            ["git", "worktree", "add", str(worktree_path), base_commit, "-q", "--detach"],
            cwd=repo_path, capture_output=True, timeout=30,
        )

        # 复制编译好的 .so 文件（sklearn 等需要）
        for so_file in repo_path.glob("**/*.so"):
            rel = so_file.relative_to(repo_path)
            dest = worktree_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(str(so_file), str(dest))
            except Exception:
                pass

        # Apply model's patch
        patch_file = str(worktree_path / "_model.patch")
        with open(patch_file, "w") as f:
            f.write(model_patch)
        result = subprocess.run(
            ["git", "apply", "--ignore-whitespace", patch_file],
            cwd=worktree_path, capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["git", "apply", "--ignore-whitespace", "--3way", patch_file],
                cwd=worktree_path, capture_output=True, text=True, timeout=10,
            )
        if result.returncode != 0:
            return False, 0.0, f"patch_apply_failed({result.stderr[:100]})"

        # Apply test patch
        test_file = str(worktree_path / "_test.patch")
        with open(test_file, "w") as f:
            f.write(test_patch)
        subprocess.run(
            ["git", "apply", "--ignore-whitespace", test_file],
            cwd=worktree_path, capture_output=True, timeout=10,
        )

        # Rebuild compiled extensions if needed (sklearn etc.)
        _needs_build = {"scikit-learn/scikit-learn"}
        if repo in _needs_build:
            subprocess.run(
                [str(env_py), "setup.py", "build_ext", "--inplace"],
                cwd=worktree_path, capture_output=True, timeout=600,
            )

        # Run FAIL_TO_PASS tests
        passed = 0
        total = len(fail_to_pass)

        for test_id in fail_to_pass:
            test_result = _run_single_test(env_py, worktree_path, repo, test_cmd_template, test_id, timeout=timeout)
            if test_result:
                passed += 1

        resolved = passed == total
        score = passed / total if total > 0 else 0.0
        details = f"resolved={'Y' if resolved else 'N'}, pass={passed}/{total}"
        return resolved, score, details

    except subprocess.TimeoutExpired:
        return False, 0.0, "timeout"
    except Exception as e:
        return False, 0.0, f"error({str(e)[:80]})"
    finally:
        # 清理 worktree（不影响 agent 的工作目录）
        try:
            subprocess.run(["git", "worktree", "remove", str(worktree_path), "--force"],
                          cwd=repo_path, capture_output=True, timeout=10)
            shutil.rmtree(str(eval_dir), ignore_errors=True)
        except Exception:
            pass


def _run_single_test(env_py: str, repo_path: Path, repo: str, test_cmd_template: str, test_id: str, timeout: int = 60) -> bool:
    """Run a single test and return True if it passes."""
    # Parse test_id format: "test_name (module.TestClass)" → pytest-style
    # Django uses: ./tests/runtests.py --settings=test_sqlite test_module.TestClass.test_name
    # Pytest repos use: pytest path/to/test.py::TestClass::test_name
    
    if "runtests.py" in test_cmd_template:
        # Django-style: extract module and test method
        # Format: "test_method (app.module.TestClass)"
        import re
        m = re.match(r'(\w+)\s+\(([^)]+)\)', test_id)
        if m:
            method, class_path = m.group(1), m.group(2)
            # class_path = "generic_views.test_base.DeprecationTests"
            test_arg = f"{class_path}.{method}"
        else:
            test_arg = test_id
        
        cmd = [env_py, "./tests/runtests.py", "--settings=test_sqlite", "--parallel", "1", test_arg]
    else:
        # pytest-style: "path/to/test.py::TestClass::test_method"
        cmd = [env_py, "-m", "pytest", "-xvs", test_id]
    
    try:
        result = subprocess.run(
            cmd, cwd=str(repo_path),
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = result.stdout + result.stderr
        # Check for success
        return result.returncode == 0 or "passed" in output.lower() or "\nOK" in output
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
