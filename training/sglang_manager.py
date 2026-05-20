"""SGLang supervisor lifecycle manager.

v10 迁移: 把 SGLang supervisor 从独立 nohup 进程改为 trainer 的 mp.Process 子进程.
好处:
  - multiprocessing authkey 共享 → /load_lora_adapter_from_tensors 能工作
  - 进程树同族 → Process.terminate() 干净结束, 无需 pkill
  - 子进程 ready/alive 可直接 poll from parent

用法 (在 trainer __init__ 或 train() 开头):
    from training.sglang_manager import SGLangSupervisorManager
    self.sglang_mgr = SGLangSupervisorManager(
        model_path="/path/to/local/model",
        port=8005,
        api_key=os.environ.get("SGLANG_API_KEY", "EMPTY"),
        gpu_id=0,
    )
    self.sglang_mgr.start()   # blocking until /v1/models responds
    # ... training ...
    self.sglang_mgr.stop()    # at shutdown

重启 (_sync_lora_to_vllm 失败时):
    self.sglang_mgr.restart()
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# 全局共享 authkey —— parent 设置一次, 所有 spawn 的子进程继承.
# spawn context 会通过 pickle 把 current_process().authkey 传给 child.
_SHARED_AUTHKEY = b"skillflow_sglang_v10_authkey_fixed_bytes_32b"


def _set_shared_authkey():
    """在 trainer 初期调一次; 之后 spawn 的所有子进程会继承."""
    mp.current_process().authkey = _SHARED_AUTHKEY


def _sglang_worker_entry(server_args_dict: dict, ready_file: str):
    """SGLang 子进程入口. 在新 Python interpreter 中执行.

    Args:
        server_args_dict: dict of ServerArgs fields (serializable across pickle).
        ready_file: path to touch when server is ready (parent polls this).
    """
    # 子进程中显式 reassert authkey (spawn 应该自动传, 但保险起见)
    import multiprocessing as _mp
    _mp.current_process().authkey = _SHARED_AUTHKEY

    # 环境: CUDA_VISIBLE_DEVICES + NO_PROXY + 减少 SGLang 噪音
    # CUDA_VISIBLE_DEVICES 已在父进程 spawn 前 set 到 os.environ 里, 子进程继承
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 延迟 import: 在子进程 CUDA context 建立前不触及 torch.cuda
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.entrypoints.http_server import launch_server

    server_args = ServerArgs(**server_args_dict)
    # launch_server 会阻塞在 uvicorn 主循环. ready 检测由父进程通过 HTTP poll 做.
    # 这里我们通过 touch ready_file 示意 "入口已执行"; 但真正的 ready 由 /v1/models 返回 200 确认.
    try:
        with open(ready_file, "w") as f:
            f.write(f"sglang worker started at {time.time()}\n")
    except Exception:
        pass

    launch_server(server_args)


class SGLangSupervisorManager:
    """Spawn & manage the SGLang supervisor as a child process of the trainer."""

    def __init__(
        self,
        model_path: str,
        port: int,
        api_key: str,
        gpu_id: int = 0,
        max_lora_rank: int = 64,
        lora_target_modules: Optional[list] = None,
        max_loras_per_batch: int = 1,
        max_loaded_loras: int = 2,
        mem_fraction_static: float = 0.82,
        context_length: int = 32768,
        served_model_name: str = "supervisor_theta",
        reasoning_parser: str = "qwen3",
        tool_call_parser: str = "qwen3_coder",
        ready_timeout_s: int = 300,
    ):
        self.model_path = model_path
        self.port = port
        self.api_key = api_key
        self.gpu_id = gpu_id
        self.max_lora_rank = max_lora_rank
        self.lora_target_modules = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.max_loras_per_batch = max_loras_per_batch
        self.max_loaded_loras = max_loaded_loras
        self.mem_fraction_static = mem_fraction_static
        self.context_length = context_length
        self.served_model_name = served_model_name
        self.reasoning_parser = reasoning_parser
        self.tool_call_parser = tool_call_parser
        self.ready_timeout_s = ready_timeout_s

        self._proc: Optional[mp.Process] = None
        self._ready_file: Optional[str] = None
        self._ctx = mp.get_context("spawn")

    # ------------ public API ------------
    def start(self) -> None:
        """Spawn SGLang child process and wait until /v1/models responds."""
        _set_shared_authkey()  # ensure parent has fixed authkey before spawn

        # Pin GPU via env *before* spawn so child inherits correct CUDA_VISIBLE_DEVICES.
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # Temp file as a best-effort "worker process started" marker.
        import tempfile
        fd, self._ready_file = tempfile.mkstemp(prefix="sglang_ready_", suffix=".txt")
        os.close(fd)

        server_args_dict = self._build_server_args_dict()

        self._proc = self._ctx.Process(
            target=_sglang_worker_entry,
            args=(server_args_dict, self._ready_file),
            name=f"SGLang-supervisor-gpu{self.gpu_id}",
            daemon=False,
        )
        self._proc.start()
        logger.info(
            f"[SGLangManager] spawned SGLang child PID={self._proc.pid} "
            f"GPU={self.gpu_id} port={self.port}"
        )

        # Restore parent's CUDA_VISIBLE_DEVICES (child already inherited its value).
        if prev_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd

        self._wait_ready()

    def stop(self, timeout_s: int = 30) -> None:
        """Terminate SGLang child + walk subtree + SIGKILL all descendants.

        Why: SGLang's launch_server spawns internal subprocesses (scheduler,
        tp_worker, detokenizer) which own CUDA contexts. If the top-level
        mp.Process dies but these grandchildren survive → GPU memory leak.
        So we must recursively kill the entire descendant tree.
        """
        if self._proc is None:
            return
        root_pid = self._proc.pid
        logger.info(f"[SGLangManager] stopping SGLang subtree rooted at PID={root_pid}")

        # 1. Collect ALL descendants (SGLang grandchildren hold CUDA)
        import subprocess as _sp
        descendants = set()
        def _collect_descendants(pid: int):
            try:
                out = _sp.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=3)
                for line in out.stdout.strip().split("\n"):
                    if line and line.isdigit():
                        cpid = int(line)
                        descendants.add(cpid)
                        _collect_descendants(cpid)
            except Exception:
                pass
        if root_pid:
            _collect_descendants(root_pid)
        logger.info(f"[SGLangManager]   descendants to kill: {sorted(descendants)}")

        # 2. SIGTERM the top-level mp.Process (lets launch_server do graceful drain)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=timeout_s)
            if self._proc.is_alive():
                logger.warning(f"[SGLangManager] SIGTERM timeout; SIGKILL top-level")
                self._proc.kill()
                self._proc.join(timeout=5)

        # 3. SIGKILL any surviving descendants (SGLang grandchildren)
        killed = 0
        for pid in descendants:
            try:
                os.kill(pid, 9)
                killed += 1
            except ProcessLookupError:
                pass  # already gone, good
            except Exception as e:
                logger.warning(f"[SGLangManager]   failed to kill PID {pid}: {e}")
        logger.info(f"[SGLangManager]   force-killed {killed} descendants")

        # 4. Safety-net: sweep for any remaining sglang:: processes with CUDA_VISIBLE_DEVICES=<our gpu>
        try:
            r = _sp.run(
                "ps aux | grep -E 'sglang::' | grep -v grep | awk '{print $2}'",
                shell=True, capture_output=True, text=True, timeout=5,
            )
            for p in r.stdout.strip().split("\n"):
                if not p.isdigit():
                    continue
                try:
                    env_path = f"/proc/{p}/environ"
                    with open(env_path, "rb") as f:
                        env = f.read().decode("utf-8", errors="ignore")
                    # Only kill if it belongs to our GPU (avoid killing unrelated sglang on other GPUs)
                    if f"CUDA_VISIBLE_DEVICES={self.gpu_id}" in env:
                        os.kill(int(p), 9)
                        logger.info(f"[SGLangManager]   swept orphan sglang:: PID {p}")
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[SGLangManager]   safety-net sweep failed: {e}")

        self._proc = None
        if self._ready_file and os.path.exists(self._ready_file):
            try:
                os.remove(self._ready_file)
            except Exception:
                pass

    def restart(self) -> None:
        """Stop current child and spawn a fresh one. Blocks until ready."""
        logger.info("[SGLangManager] restart requested")
        self.stop(timeout_s=30)
        # Wait for GPU memory to free
        self._wait_gpu_free(max_wait_s=60)
        self.start()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    # ------------ internals ------------
    def _build_server_args_dict(self) -> dict:
        """Build ServerArgs keyword args for sglang.srt.server_args.ServerArgs(**)."""
        return dict(
            model_path=self.model_path,
            port=self.port,
            api_key=self.api_key,
            reasoning_parser=self.reasoning_parser,
            tool_call_parser=self.tool_call_parser,
            trust_remote_code=True,
            served_model_name=self.served_model_name,
            context_length=self.context_length,
            mem_fraction_static=self.mem_fraction_static,
            enable_lora=True,
            max_lora_rank=self.max_lora_rank,
            lora_target_modules=self.lora_target_modules,
            max_loras_per_batch=self.max_loras_per_batch,
            max_loaded_loras=self.max_loaded_loras,
        )

    def _wait_ready(self) -> None:
        """Poll /v1/models until HTTP 200 or timeout."""
        url = f"http://127.0.0.1:{self.port}/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.time()
        while time.time() - t0 < self.ready_timeout_s:
            if self._proc is None or not self._proc.is_alive():
                raise RuntimeError(
                    f"[SGLangManager] child died early (exit={self._proc.exitcode if self._proc else 'none'})"
                )
            try:
                r = requests.get(url, headers=headers, timeout=3)
                if r.status_code == 200 and self.served_model_name in r.text:
                    elapsed = time.time() - t0
                    logger.info(f"[SGLangManager] SGLang ready after {elapsed:.1f}s")
                    # extra 5s for scheduler warmup
                    time.sleep(5)
                    return
            except Exception:
                pass
            time.sleep(3)
        raise TimeoutError(
            f"[SGLangManager] SGLang did not become ready within {self.ready_timeout_s}s"
        )

    def _wait_gpu_free(self, max_wait_s: int = 60) -> None:
        """Wait for GPU <gpu_id> memory to free below 500MB."""
        import subprocess
        t0 = time.time()
        while time.time() - t0 < max_wait_s:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits", f"--id={self.gpu_id}"],
                    capture_output=True, text=True, timeout=5,
                )
                mem = int(out.stdout.strip())
                if mem < 500:
                    return
            except Exception:
                pass
            time.sleep(2)
        logger.warning(f"[SGLangManager] GPU {self.gpu_id} still busy after {max_wait_s}s — continuing anyway")
