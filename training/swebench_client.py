

import hashlib
import logging
import os
import threading
import re
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_API = "http://35.225.163.1"
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_MODEL_NAME = "skillflow"


def _parse_stdout(stdout: str) -> Dict[str, int]:

    result = {"resolved": 0, "unresolved": 0, "error": 0, "total": 0, "empty": 0}
    for line in stdout.split("\n"):
        m = re.search(r"Instances resolved:\s*(\d+)", line)
        if m:
            result["resolved"] = int(m.group(1))
            continue
        m = re.search(r"Instances unresolved:\s*(\d+)", line)
        if m:
            result["unresolved"] = int(m.group(1))
            continue
        m = re.search(r"Instances submitted:\s*(\d+)", line)
        if m:
            result["total"] = int(m.group(1))
            continue
        m = re.search(r"Instances with errors:\s*(\d+)", line)
        if m:
            result["error"] = int(m.group(1))
            continue
        m = re.search(r"Instances with empty patches:\s*(\d+)", line)
        if m:
            result["empty"] = int(m.group(1))
    return result


class SWEBenchEvalClient:
    def __init__(
        self,
        api_url: str = DEFAULT_API,
        dataset: str = DEFAULT_DATASET,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.api_url = api_url.rstrip("/")
        self.dataset = dataset
        self.model_name = model_name
        self._cache: Dict[str, float] = {}
        self._cache_lock = threading.Lock()


        self._thread_local = threading.local()
        self.single_eval_max_workers = self._coerce_int_env(
            "SKILLFLOW_SWE_SERVER_MAX_WORKERS",
            self._coerce_int_env("SWE_SERVER_MAX_WORKERS", 4),
        )
        self.print_monitor = os.environ.get("SWE_CLIENT_PRINT", "1").lower() not in {
            "0", "false", "no", "off"
        }

    @staticmethod
    def _coerce_int_env(name: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(name, str(default))))
        except Exception:
            return default

    def _monitor(self, message: str, level: str = "info") -> None:

        if level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
        if self.print_monitor:
            print(message, flush=True)

    def health(self) -> bool:

        try:
            resp = self._session().get(f"{self.api_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def queue(self) -> Dict:

        try:
            resp = self._session().get(f"{self.api_url}/queue", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}

    def _session(self):
        sess = getattr(self._thread_local, "sess", None)
        if sess is None:
            sess = requests.Session()
            sess.trust_env = False
            self._thread_local.sess = sess
        return sess

    def _submit_batch(self, predictions: List[Dict], max_workers: int = 10) -> Optional[str]:

        try:
            resp = self._session().post(
                f"{self.api_url}/evaluate",
                json={
                    "predictions": predictions,
                    "max_workers": max_workers,
                    "dataset": self.dataset,
                },
                timeout=60,
            )
            if resp.status_code not in (200, 202):
                logger.warning(f"[SWE-client] submit HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            data = resp.json()
            task_id = data.get("task_id")
            if task_id:
                self._monitor(
                    f"[SWE-client] submitted {len(predictions)} prediction(s) "
                    f"task={task_id[:12]} max_workers={max_workers}"
                )
            return task_id
        except Exception as e:
            logger.warning(f"[SWE-client] submit error: {e}")
            return None

    def _fetch_result(self, task_id: str, timeout: int) -> Optional[Dict]:

        deadline = time.time() + timeout
        poll_interval = 5  
        first_poll = True

        while time.time() < deadline:
            try:

                remaining = deadline - time.time()
                server_wait = min(poll_interval * 4, max(int(remaining), 2))
                resp = self._session().get(
                    f"{self.api_url}/result/{task_id}",
                    params={"timeout": server_wait},
                    timeout=server_wait + 10,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("state") == "FAILURE":
                        logger.warning(
                            f"[SWE-client] task {task_id[:12]} FAILURE: {data.get('error','')[:100]}"
                        )
                        return None
                    return data.get("result")
                elif resp.status_code == 202:

                    if first_poll:
                        first_poll = False
                        logger.debug(f"[SWE-client] task {task_id[:12]} still running (202), polling...")
                    time.sleep(poll_interval)
                    continue
                else:
                    logger.warning(f"[SWE-client] result HTTP {resp.status_code}: {resp.text[:150]}")
                    return None

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:


                self._monitor(
                    f"[SWE-client] transient result polling error for task "
                    f"{task_id[:12]}: {type(e).__name__}: {str(e)[:120]}; retrying"
                    , level="warning"
                )
                time.sleep(poll_interval)
                continue
            except Exception as e:
                logger.warning(f"[SWE-client] result error: {e}")
                return None

        self._monitor(
            f"[SWE-client] result polling timeout after {timeout}s (task {task_id[:12]})",
            level="warning",
        )
        return None

    def evaluate(
        self,
        instance_id: str,
        model_patch: str,
        timeout: int = 900,
    ) -> Optional[float]:


        patch_hash = hashlib.md5(model_patch.encode()).hexdigest()[:10]
        cache_key = f"{instance_id}_{patch_hash}"
        with self._cache_lock:
            score = self._cache.get(cache_key)
        if score is not None:
            logger.info(
                f"[SWE-client] {instance_id}: cached={'RESOLVED' if score > 0 else 'unresolved'}"
            )
            return score


        if not model_patch.strip():
            logger.info(f"[SWE-client] {instance_id}: empty patch → 0.0")
            return 0.0


        predictions = [{
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": self.model_name,
        }]
        task_id = self._submit_batch(predictions, max_workers=self.single_eval_max_workers)
        if not task_id:
            return None


        result = self._fetch_result(task_id, timeout=timeout)
        if result is None:
            return None


        stdout = result.get("stdout_tail", "")
        parsed = _parse_stdout(stdout)
        total = parsed["total"]
        resolved = parsed["resolved"]

        if total == 0:

            logger.warning(f"[SWE-client] {instance_id}: total=0, returncode={result.get('returncode')}")
            return None

        score = float(resolved) / total
        with self._cache_lock:
            self._cache[cache_key] = score
        self._monitor(
            f"[SWE-client] {instance_id}: {'RESOLVED' if score > 0 else 'unresolved'} "
            f"({resolved}/{total}, task={task_id[:12]})"
        )
        return score

    def evaluate_batch(
        self,
        predictions: List[Dict],
        timeout: int = 600,
        max_workers: int = 10,
    ) -> Dict[str, float]:

        if not predictions:
            return {}

        task_id = self._submit_batch(predictions, max_workers=max_workers)
        if not task_id:
            return {}

        result = self._fetch_result(task_id, timeout=timeout)
        if result is None:
            return {}

        stdout = result.get("stdout_tail", "")
        parsed = _parse_stdout(stdout)

        total = parsed["total"]
        if total == 0:
            return {}
        ratio = float(parsed["resolved"]) / total

        results = {}
        for p in predictions:
            iid = p["instance_id"]
            patch = p.get("model_patch", "")
            patch_hash = hashlib.md5(patch.encode()).hexdigest()[:10]
            results[f"{iid}_{patch_hash}"] = ratio
        return results


_client: Optional[SWEBenchEvalClient] = None
_client_lock = threading.Lock()


def get_client() -> SWEBenchEvalClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = SWEBenchEvalClient()
    return _client
