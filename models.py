from typing import List, Literal
from pydantic import BaseModel, Field

# Allowed directions (STRICT)
Direction = Literal["up", "down", "left", "right", "stay"]

# Allowed events (STRICT)
EventType = Literal[
    "start",
    "moved",
    "looked",
    "hazard_detected",
    "hazard_hit",
    "goal",
    "privacy_violation",
    "timeout",
]


class BlinkAction(BaseModel):
    """Action sent by the agent each step"""

    move: Direction = Field(..., description="Movement direction")
    look: Direction = Field(..., description="Direction to look")
    inspect: bool = Field(..., description="Whether to scan for hazards")


class BlinkObservation(BaseModel):
    """Observation returned to agent (partial visibility)"""

    patch: List[List[str]] = Field(
        ..., description="3x3 visible grid around agent"
    )
    agent_pos: List[int] = Field(
        ..., description="[row, col] position of agent"
    )
    look_center: List[int] = Field(
        ..., description="[row, col] position where agent is looking"
    )
    step_count: int = Field(
        ..., ge=0, description="Current step count"
    )
    max_steps: int = Field(
        ..., gt=0, description="Maximum allowed steps"
    )
    last_event: EventType = Field(
        ..., description="Last event occurred in environment"
    )


class BlinkState(BaseModel):
    """Full state visible to judges (not agent)"""

    episode_id: str = Field(
        ..., description="Unique identifier for the episode"
    )
    full_grid: List[List[str]] = Field(
        ..., description="Complete 7x7 environment grid"
    )
    agent_pos: List[int] = Field(
        ..., description="Agent position"
    )
    look_center: List[int] = Field(
        ..., description="Agent look position"
    )
    step_count: int = Field(
        ..., ge=0, description="Steps taken"
    )
    max_steps: int = Field(
        ..., gt=0, description="Max steps"
    )
    episode_reward: float = Field(
        ..., description="Total accumulated reward"
    )
    done: bool = Field(
        ..., description="Whether episode is finished"
    )
    privacy_violations: int = Field(
        ..., ge=0, description="Number of privacy violations"
    )
