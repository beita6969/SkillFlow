

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _alfworld_env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() not in {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() not in {"0", "false", "no", "off"}


def _coerce_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return bool(default)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() not in {"0", "false", "no", "off", "none", "null", ""}


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return default
    if str(val).strip().lower() in {"none", "null"}:
        return None
    return int(str(val).strip())


_SKILLRL_ALFWORLD = os.environ.get("SKILLRL_ALFWORLD_PATH", "")
_SKILLRL_WEBSHOP = os.environ.get("SKILLRL_WEBSHOP_PATH", "")
_RAGEN_ROOT = os.environ.get("RAGEN_ROOT", "")

for p in [_SKILLRL_ALFWORLD, _SKILLRL_WEBSHOP, _RAGEN_ROOT]:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)


if not os.environ.get("ALFWORLD_DATA"):
    os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")


_conda_prefix = os.environ.get("CONDA_PREFIX")
if not os.environ.get("JAVA_HOME") and _conda_prefix and os.path.isdir(os.path.join(_conda_prefix, "bin")):
    os.environ["JAVA_HOME"] = _conda_prefix
    os.environ["PATH"] = os.path.join(_conda_prefix, "bin") + ":" + os.environ.get("PATH", "")


@dataclass
class BaseEnvConfig:
    invalid_act: str = ""

@dataclass
class AlfredEnvConfig(BaseEnvConfig):
    config_file: str = os.path.join(_RAGEN_ROOT, "ragen/env/alfworld/alfworld_config.yaml")
    score: float = 10.0
    render_mode: str = "text"
    eval_dataset: str = "eval_in_distribution"


class BaseLanguageBasedEnv:

    def __init__(self):
        pass
    def get_available_actions(self):
        return []


def _check_alfworld() -> bool:
    try:
        import textworld  
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  
        return True
    except ImportError as e:
        logger.info(f"[RAGEN] ALFWorld not available: {e}")
        return False


def _check_webshop() -> bool:
    try:
        webshop_path = os.path.join(_SKILLRL_WEBSHOP, "webshop")
        if webshop_path not in sys.path:
            sys.path.append(webshop_path)  
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  
        return True
    except Exception as e:
        logger.info(f"[RAGEN] WebShop not available: {e}")
        return False


_ALFWORLD_RAW_ENV_CACHE: Dict = {}  
_WEBSHOP_INIT_LOCK = threading.Lock()


class ALFWorldEnv:


    def __init__(self, config: AlfredEnvConfig, mode: str = "train"):
        self.config = config
        self.score = config.score
        self.mode = mode
        self.alfred_env = None
        self.task_description = ""
        self.current_seed = None
        self.current_game_index = None
        self.current_game_file = ""
        self.current_task_dir = ""


        cache_key = (config.config_file, mode)
        if cache_key not in _ALFWORLD_RAW_ENV_CACHE:
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
            import yaml

            with open(config.config_file) as f:
                raw = f.read()

            raw = raw.replace("$ALFWORLD_DATA", os.environ.get("ALFWORLD_DATA", ""))
            raw_config = yaml.safe_load(raw)

            raw_env = AlfredTWEnv(config=raw_config, train_eval=mode)

            _ALFWORLD_RAW_ENV_CACHE[cache_key] = {
                "num_games": raw_env.num_games,
                "game_files": list(raw_env.game_files),
            }
            logger.info(f"[ALFWorld] Cached {raw_env.num_games} game files for mode={mode}")

        cached = _ALFWORLD_RAW_ENV_CACHE[cache_key]
        self.num_games = cached["num_games"]
        self.game_files = cached["game_files"]

    def reset(self, seed=None):
        import textworld
        import textworld.gym
        self.current_seed = seed

        if seed is not None:
            idx = seed % max(self.num_games, 1)
        else:
            import random
            idx = random.randint(0, max(self.num_games - 1, 0))

        if not self.game_files:
            raise RuntimeError("No game files found. Check ALFWORLD_DATA env var.")


        if self.alfred_env is not None:
            try:
                self.alfred_env.close()
            except Exception:
                pass
            self.alfred_env = None


        import threading
        _lock = getattr(ALFWorldEnv, '_register_lock', None)
        if _lock is None:
            ALFWorldEnv._register_lock = threading.Lock()
            _lock = ALFWorldEnv._register_lock

        max_retries = 5
        for attempt in range(max_retries):
            try_idx = (idx + attempt) % max(self.num_games, 1)
            game_file = self.game_files[try_idx]
            task_dir = os.path.basename(os.path.dirname(os.path.dirname(game_file)))
            self.current_game_index = try_idx
            self.current_game_file = game_file
            self.current_task_dir = task_dir
            self.task_description = self._parse_task_from_path(task_dir)

            try:
                request_infos = textworld.EnvInfos(
                    won=True, admissible_commands=True,
                    score=True, max_score=True,
                    description=True, inventory=True,
                    extras=["gamefile"],
                )


                try:
                    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
                    wrappers = [AlfredDemangler]
                except ImportError:
                    wrappers = []


                with _lock:
                    env_id = textworld.gym.register_game(
                        game_file,
                        request_infos=request_infos,
                        max_episode_steps=50,
                        wrappers=wrappers,
                    )
                    self.alfred_env = textworld.gym.make(env_id)
                    obs, infos = self.alfred_env.reset()

                if isinstance(obs, (list, tuple)):
                    obs = obs[0]
                available = infos.get("admissible_commands", [])
                if available and isinstance(available[0], list):
                    available = available[0]
                self._admissible_commands = available

                obs_str = str(obs)

                task_marker = "Your task is to: "
                if task_marker in obs_str:
                    self.task_description = obs_str[obs_str.find(task_marker) + len(task_marker):].strip()


                return obs_str

            except Exception as e:
                logger.warning(f"[ALFWorld] Game file {try_idx} failed (attempt {attempt+1}): {e}")
                if self.alfred_env is not None:
                    try:
                        self.alfred_env.close()
                    except Exception:
                        pass
                    self.alfred_env = None

        raise RuntimeError(f"ALFWorld: all {max_retries} game files failed")

    @staticmethod
    def _parse_task_from_path(task_dir: str) -> str:

        parts = task_dir.split("-")
        if len(parts) < 3:
            return ""
        task_type = parts[0]
        obj = parts[1]
        target = parts[3] if len(parts) > 3 else parts[2]

        _surfaces = ['countertop', 'shelf', 'desk', 'sidetable', 'bed', 'sofa', 'armchair',
                     'coffeetable', 'diningtable', 'stoveburner', 'toilet', 'bathtub', 'garbagecan', 'ottoman']
        _prep = 'on' if any(s in target.lower() for s in _surfaces) else 'in'
        task_map = {
            "pick_and_place_simple": f"put {obj} {_prep} {target}",
            "pick_clean_then_place_in_recep": f"clean {obj} and put it {_prep} {target}",
            "pick_heat_then_place_in_recep": f"heat {obj} and put it {_prep} {target}",
            "pick_cool_then_place_in_recep": f"cool {obj} and put it {_prep} {target}",
            "look_at_obj_in_light": f"examine {obj} with the desk lamp",
            "pick_two_obj_and_place": f"put two {obj}s {_prep} {target}",
        }
        return task_map.get(task_type, task_dir.replace("_", " "))

    def step(self, action):
        import re as _re


        if _alfworld_env_flag("ALFWORLD_CANONICALIZE_ACTION", False):
            _m = _re.match(r'^put\s+(.+?)\s+(?:in|on)\s+(.+)$', action, flags=_re.IGNORECASE)
            if _m:
                action = f'move {_m.group(1)} to {_m.group(2)}'

        prev_available = list(getattr(self, "_admissible_commands", []) or [])
        action_is_valid = str(action) in [str(a) for a in prev_available]


        _lock = getattr(ALFWorldEnv, '_register_lock', None)
        if _lock is None:
            import threading
            ALFWorldEnv._register_lock = threading.Lock()
            _lock = ALFWorldEnv._register_lock

        with _lock:
            obs, score, done, infos = self.alfred_env.step(action)


        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        available = infos.get("admissible_commands", [])
        if available and isinstance(available[0], list):
            available = available[0]
        self._admissible_commands = available
        won = infos.get("won", False)
        if isinstance(won, (list, tuple)):
            won = won[0] if won else False
        if isinstance(done, (list, tuple)):
            done = done[0] if done else False
        if isinstance(score, (list, tuple)):
            score = score[0] if score else 0


        obs_text = str(obs)

        reward = self.score if (done and won) else 0.0

        return obs_text, reward, done, {
            "won": won,
            "score": score,
            "available_actions": available,


            "action_is_valid": action_is_valid,
            "action_is_effective": "nothing happens" not in obs_text.strip().lower(),
        }


class WebShopEnv:


    def __init__(
        self,
        observation_mode="text",
        *,
        human_goals: bool = False,
        use_small: bool = True,
        num_products: Optional[int] = None,
        goal_split: str = "skillrl_val",
        goal_offset: Optional[int] = None,
        goal_count: Optional[int] = None,
        file_path: Optional[str] = None,
        attr_path: Optional[str] = None,
        env_seed: int = 1000,
    ):
        webshop_path = os.path.join(_SKILLRL_WEBSHOP, "webshop")
        if webshop_path not in sys.path:
            sys.path.insert(0, webshop_path)

        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        if file_path is None or attr_path is None:
            data_dir = os.path.join(webshop_path, "data")
            if use_small:
                file_path = file_path or os.path.join(data_dir, "items_shuffle_1000.json")
                attr_path = attr_path or os.path.join(data_dir, "items_ins_v2_1000.json")
            else:
                file_path = file_path or os.path.join(data_dir, "items_shuffle.json")
                attr_path = attr_path or os.path.join(data_dir, "items_ins_v2.json")


        self.env_seed = int(env_seed)
        with _WEBSHOP_INIT_LOCK:
            self.env = WebAgentTextEnv(
                observation_mode=observation_mode,
                human_goals=bool(human_goals),


                num_products=num_products,
                file_path=file_path,
                attr_path=attr_path,
                seed=self.env_seed,
            )
        self.human_goals = bool(human_goals)
        self.use_small = bool(use_small)
        self.num_products = num_products
        self.file_path = file_path
        self.attr_path = attr_path
        self.goal_split = str(goal_split or "skillrl_val")
        self._n_goals = len(getattr(self.env.server, 'goals', []) if hasattr(self.env, 'server') else []) or 0
        self.goal_indices = self._build_goal_indices(
            self._n_goals,
            split=self.goal_split,
            offset=goal_offset,
            count=goal_count,
        )
        self.current_goal_index: Optional[int] = None
        self.current_goal_instruction: str = ""
        self.reward = 0.0
        logger.info(
            "[WebShop] Initialized with human_goals=%s, use_small=%s, "
            "num_products=%s, n_goals=%s, split=%s, eval_indices=%s",
            self.human_goals, self.use_small, self.num_products, self._n_goals,
            self.goal_split, len(self.goal_indices),
        )

    @staticmethod
    def _build_goal_indices(n_goals: int, *, split: str, offset: Optional[int], count: Optional[int]) -> list[int]:
        if n_goals <= 0:
            return []
        if offset is not None or count is not None:
            start = max(0, int(offset or 0))
            stop = n_goals if count is None else min(n_goals, start + max(0, int(count)))
            return list(range(start, stop))

        split_l = str(split or "skillrl_val").strip().lower()


        if split_l in {"skillrl_val", "val", "eval", "test"}:
            return list(range(0, min(500, n_goals)))
        if split_l in {"train", "skillrl_train"}:
            if n_goals > 500:
                return list(range(500, n_goals))
            return list(range(n_goals))
        if split_l in {"all", "full"}:
            return list(range(n_goals))
        logger.warning("[WebShop] Unknown goal_split=%r; falling back to SkillRL val prefix.", split)
        return list(range(0, min(500, n_goals)))

    def reset(self, seed=None, goal_index: Optional[int] = None):
        if goal_index is not None:
            session = int(goal_index)
            if not (0 <= session < self._n_goals):
                raise IndexError(f"WebShop goal_index={session} out of range n_goals={self._n_goals}")
        elif self.goal_indices:


            session = self.goal_indices[(int(seed) if seed is not None else 0) % len(self.goal_indices)]
        else:
            session = 0
        try:
            obs, info = self.env.reset(session=session)
        except Exception:
            logger.exception("[WebShop] reset failed for session=%s; falling back to session=0", session)
            session = 0
            obs, info = self.env.reset(session=session)
        self.current_goal_index = session
        try:
            self.current_goal_instruction = str(self.env.server.goals[session].get("instruction_text", ""))
        except Exception:
            self.current_goal_instruction = ""
        info = dict(info or {})
        info["available_actions"] = self.env.get_available_actions()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})
        info["available_actions"] = self.env.get_available_actions()
        self.reward = reward


        if done:
            info["won"] = reward == 1.0  
            info["graded_score"] = reward  
            return obs, reward, True, info
        else:
            info["won"] = False
            return obs, 0.0, done, info


class RAGENAdapter:


    def __init__(self):
        self._env = None
        self._env_type: str = ""
        self._done: bool = False
        self._total_reward: float = 0.0
        self._available_actions: list = []

    @property
    def available_actions(self) -> list:

        return self._available_actions

    def reset(self, env_type: str, env_config: Dict[str, Any],
              question: str = "", extra: Optional[Dict] = None) -> str:
        self._env_type = env_type
        self._done = False
        self._total_reward = 0.0

        if env_type == "alfworld":
            return self._reset_alfworld(env_config)
        elif env_type == "webshop":
            return self._reset_webshop(env_config)
        else:
            self._env = None
            return f"[ENV_UNAVAILABLE] Unknown environment type: {env_type}"

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self._done:
            return "[Done] Episode already finished.", 0.0, True, {}
        if self._env is None:
            return "[ENV_UNAVAILABLE] No environment initialized.", 0.0, False, {}


        import re as _re
        if self._env_type == "alfworld" and _alfworld_env_flag("ALFWORLD_CANONICALIZE_ACTION", False):
            _m = _re.match(r'^put\s+(.+?)\s+(?:in|on)\s+(.+)$', action, flags=_re.IGNORECASE)
            if _m:
                action = f'move {_m.group(1)} to {_m.group(2)}'

        try:
            obs, reward, done, info = self._env.step(action)
            self._done = done
            self._total_reward += reward
            if isinstance(obs, (list, tuple)):
                obs = obs[0] if obs else ""

            self._available_actions = info.get("available_actions", info.get("admissible_commands", []))
            return str(obs), reward, done, info
        except Exception as e:
            logger.warning(f"[RAGEN] step error: {e}")
            return f"[ERROR] Environment step failed: {e}", 0.0, False, {"error": str(e)}

    def _reset_alfworld(self, config: Dict) -> str:
        if not _check_alfworld():
            self._env = None
            return "[ENV_UNAVAILABLE] ALFWorld requires textworld package."

        try:
            alf_config = AlfredEnvConfig()
            if "config_file" in config:
                alf_config.config_file = config["config_file"]
            mode = config.get("mode", "train")

            self._env = ALFWorldEnv(config=alf_config, mode=mode)
            seed = config.get("seed")
            obs = self._env.reset(seed=seed)

            self._available_actions = getattr(self._env, '_admissible_commands', [])
            return str(obs)
        except Exception as e:
            logger.warning(f"[RAGEN] ALFWorld reset failed: {e}")
            self._env = None
            return f"[ENV_UNAVAILABLE] ALFWorld init failed: {e}"

    def _reset_webshop(self, config: Dict) -> str:
        if not _check_webshop():
            self._env = None
            return "[ENV_UNAVAILABLE] WebShop requires gym + web_agent_site."

        try:
            obs_mode = config.get("observation_mode", "text")
            human_goals = _coerce_bool(config.get("human_goals", None), _env_bool("WEBSHOP_HUMAN_GOALS", False))
            use_small = _coerce_bool(config.get("use_small", None), _env_bool("WEBSHOP_USE_SMALL", True))
            num_products = config.get("num_products", _env_int("WEBSHOP_NUM_PRODUCTS", None))
            goal_split = str(config.get("goal_split", os.environ.get("WEBSHOP_GOAL_SPLIT", "skillrl_val")))
            goal_offset = config.get("goal_offset", _env_int("WEBSHOP_GOAL_OFFSET", None))
            goal_count = config.get("goal_count", _env_int("WEBSHOP_GOAL_COUNT", None))
            file_path = config.get("file_path") or os.environ.get("WEBSHOP_FILE_PATH") or None
            attr_path = config.get("attr_path") or os.environ.get("WEBSHOP_ATTR_PATH") or None
            env_seed = config.get("env_seed", _env_int("WEBSHOP_ENV_SEED", 1000))
            self._env = WebShopEnv(
                observation_mode=obs_mode,
                human_goals=human_goals,
                use_small=use_small,
                num_products=num_products,
                goal_split=goal_split,
                goal_offset=goal_offset,
                goal_count=goal_count,
                file_path=file_path,
                attr_path=attr_path,
                env_seed=int(env_seed if env_seed is not None else 1000),
            )
            seed = config.get("seed")
            goal_index = config.get("goal_index")
            obs = self._env.reset(seed=seed, goal_index=goal_index)

            self._available_actions = self._env.env.get_available_actions()
            return str(obs)
        except Exception as e:
            logger.warning(f"[RAGEN] WebShop reset failed: {e}")
            self._env = None
            return f"[ENV_UNAVAILABLE] WebShop init failed: {e}"
