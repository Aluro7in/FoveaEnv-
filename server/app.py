import os
import sys

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import FoveaEnv
from grader import grade_episode
from models import BlinkAction

VALID_TASKS = ["easy", "medium", "hard"]
VALID_DIRECTIONS = ["up", "down", "left", "right", "stay"]

app = FastAPI(
    title="FoveaEnv",
    description="Privacy-aware attention navigation benchmark",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = FoveaEnv()


def _with_observation(data: dict) -> dict:
    response = dict(data)
    response["observation"] = data
    response["info"] = {}
    return response


def _score_payload(last_event: str) -> dict:
    state = env.state()
    reached_goal = bool(getattr(env, "goal_reached", False) or last_event == "goal")
    score = grade_episode(
        episode_reward=state.episode_reward,
        reached_goal=reached_goal,
        privacy_violations=state.privacy_violations,
        total_steps=state.step_count,
    )
    return {**score, "reached_goal": reached_goal}


@app.get("/")
def root():
    return {"name": "FoveaEnv", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: dict = Body(default={})):
    task_id = req.get("task_id", "easy")
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=400, detail="Invalid task_id")

    obs = env.reset(task_id)
    return _with_observation(obs.model_dump())


@app.post("/step")
def step(req: dict = Body(default={})):
    move = req.get("move", "stay")
    look = req.get("look", "stay")
    inspect = req.get("inspect", False)

    if move not in VALID_DIRECTIONS:
        raise HTTPException(status_code=400, detail="Invalid move")
    if look not in VALID_DIRECTIONS:
        raise HTTPException(status_code=400, detail="Invalid look")

    action = BlinkAction(move=move, look=look, inspect=inspect)

    try:
        obs, reward, done = env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = obs.model_dump()
    score = _score_payload(data.get("last_event", "moved"))
    response = _with_observation(data)
    response.update(
        {
            "reward": reward,
            "done": done,
            "truncated": False,
            "score": score,
            "final_score": score["final_score"],
            "navigation_score": score["navigation_score"],
            "privacy_efficiency_score": score["privacy_efficiency_score"],
            "reached_goal": score["reached_goal"],
        }
    )
    return response


@app.get("/state")
def state():
    return env.state().model_dump()


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
