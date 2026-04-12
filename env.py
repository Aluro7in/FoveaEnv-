# env.py
# FoveaEnv — Core Environment Logic (FINAL VERSION)

import uuid
from typing import List

from models import BlinkAction, BlinkObservation, BlinkState
from tasks import get_task


class FoveaEnv:

    def __init__(self):
        self.grid = None
        self.agent_pos = [0, 0]
        self.look_center = [0, 0]
        self.step_count = 0
        self.max_steps = 40
        self.episode_reward = 0.0
        self.done = False
        self.privacy_violations = 0
        self.goal_reached = False
        self.episode_id = None
        self.goal_pos = None  # optimization

    # ─────────────────────────────────────────
    # API 1: reset()
    # ─────────────────────────────────────────
    def reset(self, task_id: str = "easy") -> BlinkObservation:
        task = get_task(task_id)

        self.grid = task["map"]
        self.max_steps = task["max_steps"]

        self.step_count = 0
        self.episode_reward = 0.0
        self.done = False
        self.privacy_violations = 0
        self.goal_reached = False

        self.episode_id = str(uuid.uuid4())

        self.agent_pos = self._find_cell('S')
        self.look_center = self.agent_pos.copy()
        self.goal_pos = self._find_cell('G')  # cache goal

        return self._make_observation("start")

    # ─────────────────────────────────────────
    # API 2: step()
    # ─────────────────────────────────────────
    def step(self, action: BlinkAction):
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")
            
        self.step_count += 1
        
        reward = -0.01  # step cost
        event = "moved"

        # ───── 1. LOOK ─────
        if action.look != "stay":
            self.look_center = self._try_move(self.look_center, action.look)
            reward -= 0.03
            event = "looked"

        # ───── 2. PRIVACY CHECK (after look) ─────
        patch = self._extract_patch(self.look_center)
        if any('P' in row for row in patch):
            reward -= 0.1
            self.privacy_violations += 1
            event = "privacy_violation"

        # ───── 3. INSPECT ─────
        if action.inspect:
            nearby = self._extract_patch(self.agent_pos)
            if any('H' in row for row in nearby):
                reward += 0.2
                event = "hazard_detected"

        # ───── 4. MOVE ─────
        if action.move != "stay":
            new_pos = self._try_move(self.agent_pos, action.move)
            cell = self.grid[new_pos[0]][new_pos[1]]

            if cell == 'H':
                reward -= 0.5
                event = "hazard_hit"

            elif cell == 'G':
                self.agent_pos = new_pos
                reward += 1.0
                self.done = True
                self.goal_reached = True
                event = "goal"

            else:
                old_dist = self._dist_to_goal(self.agent_pos)
                new_dist = self._dist_to_goal(new_pos)

                if new_dist < old_dist:
                    reward += 0.05

                self.agent_pos = new_pos

        # ───── 5. STEP + TIMEOUT ────
        
        if self.step_count >= self.max_steps and not self.done:
            reward -= 0.3
            self.done = True
            event = "timeout"

        self.episode_reward += reward

        obs = self._make_observation(event)

        return obs, round(reward, 4), self.done

    # ─────────────────────────────────────────
    # API 3: state()
    # ─────────────────────────────────────────
    def state(self) -> BlinkState:
        return BlinkState(
            episode_id=self.episode_id,
            full_grid=self.grid,
            agent_pos=self.agent_pos.copy(),
            look_center=self.look_center.copy(),
            step_count=self.step_count,
            max_steps=self.max_steps,
            episode_reward=round(self.episode_reward, 4),
            done=self.done,
            privacy_violations=self.privacy_violations
        )

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────
    def _make_observation(self, event: str) -> BlinkObservation:
        return BlinkObservation(
            patch=self._extract_patch(self.look_center),
            agent_pos=self.agent_pos.copy(),
            look_center=self.look_center.copy(),
            step_count=self.step_count,
            max_steps=self.max_steps,
            last_event=event
        )

    def _extract_patch(self, center: List[int]) -> List[List[str]]:
        r, c = center
        rows, cols = len(self.grid), len(self.grid[0])

        patch = []
        for dr in [-1, 0, 1]:
            row = []
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    row.append(self.grid[nr][nc])
                else:
                    row.append('#')
            patch.append(row)

        return patch

    def _try_move(self, pos: List[int], direction: str) -> List[int]:
        MOVES = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
            "stay": (0, 0)
        }

        dr, dc = MOVES.get(direction, (0, 0))

        r = max(0, min(len(self.grid) - 1, pos[0] + dr))
        c = max(0, min(len(self.grid[0]) - 1, pos[1] + dc))

        return [r, c]

    def _find_cell(self, cell_type: str) -> List[int]:
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == cell_type:
                    return [r, c]
        raise ValueError(f"Cell '{cell_type}' not found")

    def _dist_to_goal(self, pos: List[int]) -> int:
        return abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1])
