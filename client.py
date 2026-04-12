# client.py
# FoveaEnv HTTP client
# Matches the current server/app.py contract:
#   GET  /health
#   POST /reset
#   POST /step
#   GET  /state
#
# Returns real Pydantic models instead of flattening observations.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, Type, TypeVar

import requests

from models import BlinkAction, BlinkObservation, BlinkState

ObsT = TypeVar("ObsT")

VALID_TASKS = ("easy", "medium", "hard")
VALID_DIRECTIONS = ("up", "down", "left", "right", "stay")
DEFAULT_URL = "http://localhost:7860"


@dataclass
class StepResult(Generic[ObsT]):
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class FoveaEnvClient:
    """
    HTTP client for FoveaEnv.

    Example:
        client = FoveaEnvClient("http://localhost:7860")
        result = client.reset("easy")
        step = client.step(move="right", look="stay", inspect=False)
        state = client.state()
        client.close()
    """

    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._last_score: Optional[Dict[str, Any]] = None

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "FoveaEnvClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def last_score(self) -> Optional[Dict[str, Any]]:
        return self._last_score

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self._url(path)
        try:
            response = self._session.request(
                method=method,
                url=url,
                json=json,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Unexpected non-JSON-object response from {path}")
            return data
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Cannot reach FoveaEnv server at {url}. "
                f"Check that the server is running on {self.base_url}."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            message = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"Server error at {path}: {message}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Invalid JSON response from {path}") from exc

    @staticmethod
    def _model_validate(model_cls: Type[Any], payload: Dict[str, Any]) -> Any:
        if hasattr(model_cls, "model_validate"):
            return model_cls.model_validate(payload)  # Pydantic v2
        return model_cls.parse_obj(payload)  # Pydantic v1 fallback

    @staticmethod
    def _model_dump(model: Any) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump(exclude_none=True)
        if hasattr(model, "dict"):
            return model.dict(exclude_none=True)
        raise TypeError(f"Unsupported model type: {type(model)!r}")

    @staticmethod
    def _extract_observation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        obs = payload.get("observation")
        if isinstance(obs, dict):
            return obs
        ignored = {
            "reward",
            "done",
            "truncated",
            "score",
            "final_score",
            "navigation_score",
            "privacy_efficiency_score",
            "reached_goal",
            "info",
        }
        return {k: v for k, v in payload.items() if k not in ignored}

    def ping(self) -> bool:
        try:
            data = self._request("GET", "/health")
            return data.get("status") == "ok"
        except Exception:
            try:
                data = self._request("GET", "/")
                return data.get("status") == "running"
            except Exception:
                return False

    def reset(self, task_id: str = "easy") -> StepResult[BlinkObservation]:
        if task_id not in VALID_TASKS:
            raise ValueError(
                f"task_id must be one of {VALID_TASKS}. Got: {task_id!r}"
            )

        self._last_score = None
        payload = self._request("POST", "/reset", json={"task_id": task_id})
        obs_payload = self._extract_observation_payload(payload)
        observation = self._model_validate(BlinkObservation, obs_payload)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
            info=dict(payload.get("info", {})),
            raw=payload,
        )

    def step(
        self,
        action: Optional[BlinkAction] = None,
        *,
        move: str = "stay",
        look: str = "stay",
        inspect: bool = False,
    ) -> StepResult[BlinkObservation]:
        """
        Step the environment.

        Supports either:
            client.step(BlinkAction(...))
        or:
            client.step(move="right", look="stay", inspect=False)
        """
        if action is None:
            if move not in VALID_DIRECTIONS:
                raise ValueError(f"move must be one of {VALID_DIRECTIONS}. Got: {move!r}")
            if look not in VALID_DIRECTIONS:
                raise ValueError(f"look must be one of {VALID_DIRECTIONS}. Got: {look!r}")
            action = BlinkAction(move=move, look=look, inspect=inspect)
        elif isinstance(action, dict):
            action = BlinkAction(**action)
        elif not isinstance(action, BlinkAction):
            raise TypeError("action must be BlinkAction, dict, or None")

        payload = self._model_dump(action)
        response = self._request("POST", "/step", json=payload)

        obs_payload = self._extract_observation_payload(response)
        observation = self._model_validate(BlinkObservation, obs_payload)

        info = dict(response.get("info", {}))
        for key in (
            "score",
            "final_score",
            "navigation_score",
            "privacy_efficiency_score",
            "reached_goal",
            "truncated",
        ):
            if key in response and key not in info:
                info[key] = response[key]

        if response.get("done") and "score" in response:
            self._last_score = response["score"]
            info["score"] = response["score"]

        return StepResult(
            observation=observation,
            reward=response.get("reward"),
            done=bool(response.get("done", False)),
            info=info,
            raw=response,
        )

    def state(self) -> BlinkState:
        payload = self._request("GET", "/state")
        return self._model_validate(BlinkState, payload)


if __name__ == "__main__":
    client = FoveaEnvClient()

    print(f"Pinging {client.base_url} ...", end=" ")
    if not client.ping():
        print("FAILED")
        raise SystemExit(1)
    print("OK")

    for task in VALID_TASKS:
        result = client.reset(task)
        print(f"[{task}] reset: step_count={result.observation.step_count}, done={result.done}")

        step_result = client.step(move="right", look="stay", inspect=False)
        print(
            f"       step: reward={step_result.reward}, done={step_result.done}, "
            f"event={step_result.observation.last_event}"
        )

        state = client.state()
        print(
            f"       state: episode_id={state.episode_id}, "
            f"steps={state.step_count}, reward={state.episode_reward}"
        )

    client.close()
