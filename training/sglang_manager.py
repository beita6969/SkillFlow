
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


_SHARED_AUTHKEY = b"skillflow_sglang_v10_authkey_fixed_bytes_32b"


def _set_shared_authkey():

    mp.current_process().authkey = _SHARED_AUTHKEY


def _sglang_worker_entry(server_args_dict: dict, ready_file: str):


    import multiprocessing as _mp
    _mp.current_process().authkey = _SHARED_AUTHKEY


    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


    from sglang.srt.server_args import ServerArgs
    from sglang.srt.entrypoints.http_server import launch_server

    server_args = ServerArgs(**server_args_dict)


    try:
        with open(ready_file, "w") as f:
            f.write(f"sglang worker started at {time.time()}\n")
    except Exception:
        pass

    launch_server(server_args)


class SGLangSupervisorManager:


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


    def start(self) -> None:

        _set_shared_authkey()  


        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)


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


        if prev_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd

        self._wait_ready()

    def stop(self, timeout_s: int = 30) -> None:

        if self._proc is None:
            return
        root_pid = self._proc.pid
        logger.info(f"[SGLangManager] stopping SGLang subtree rooted at PID={root_pid}")


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


        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=timeout_s)
            if self._proc.is_alive():
                logger.warning(f"[SGLangManager] SIGTERM timeout; SIGKILL top-level")
                self._proc.kill()
                self._proc.join(timeout=5)


        killed = 0
        for pid in descendants:
            try:
                os.kill(pid, 9)
                killed += 1
            except ProcessLookupError:
                pass  
            except Exception as e:
                logger.warning(f"[SGLangManager]   failed to kill PID {pid}: {e}")
        logger.info(f"[SGLangManager]   force-killed {killed} descendants")


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

        logger.info("[SGLangManager] restart requested")
        self.stop(timeout_s=30)

        self._wait_gpu_free(max_wait_s=60)
        self.start()

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None


    def _build_server_args_dict(self) -> dict:

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

                    time.sleep(5)
                    return
            except Exception:
                pass
            time.sleep(3)
        raise TimeoutError(
            f"[SGLangManager] SGLang did not become ready within {self.ready_timeout_s}s"
        )

    def _wait_gpu_free(self, max_wait_s: int = 60) -> None:

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
