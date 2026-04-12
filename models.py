# models.py
# Data schemas for FoveaEnv — foundation of the entire project

from pydantic import BaseModel
from typing import List


class BlinkAction(BaseModel):
    """What the AI agent SENDS to the environment each step"""
    move: str    # "up" / "down" / "left" / "right" / "stay"
    look: str    # "up" / "down" / "left" / "right" / "stay"
    inspect: bool  # True = scan surroundings for hazards


class BlinkObservation(BaseModel):
    """What the agent RECEIVES after each step — PARTIAL info only"""
    patch: List[List[str]]  # 3x3 visible grid (limited vision)
    agent_pos: List[int]    # [row, col] — where agent is
    look_center: List[int]  # [row, col] — where agent is looking
    step_count: int         # how many steps taken so far
    max_steps: int          # episode step limit
    last_event: str         # what happened: "moved"/"hazard_hit"/"goal"/etc.


class BlinkState(BaseModel):
    """What JUDGES see via state() — FULL hidden information"""
    full_grid: List[List[str]]  # complete 7x7 map revealed
    agent_pos: List[int]
    look_center: List[int]
    step_count: int
    max_steps: int
    episode_reward: float       # total reward accumulated
    done: bool                  # is episode finished?
    privacy_violations: int     # how many times agent looked at P zones