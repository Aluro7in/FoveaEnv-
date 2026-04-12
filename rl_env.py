import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env import FoveaEnv
from models import BlinkAction

CHAR_MAP = {
    '#': 0,
    '.': 1,
    'H': 2,
    'P': 3,
    'S': 4,
    'G': 5,
}

MOVES = ["up", "down", "left", "right", "stay"]
ACTION_MAP = [
    (move, look, inspect)
    for move in MOVES
    for look in MOVES
    for inspect in [False, True]
]


class ForveaGymEnv(gym.Env):
    """Gym wrapper for the FoveaEnv environment."""

    def __init__(self, task_id="medium"):
        super().__init__()
        self.task_id = task_id
        self._env = None
        self.action_space = spaces.Discrete(len(ACTION_MAP))
        self.observation_space = spaces.Box(
            low=0,
            high=6,
            shape=(14,),
            dtype=np.int8,
        )

    def reset(self, seed=None, _options=None):
        if seed is not None:
            np.random.seed(seed)
        self._env = FoveaEnv()
        obs = self._env.reset(self.task_id)
        return self._encode_observation(obs), {}

    def step(self, action):
        move, look, inspect = ACTION_MAP[int(action)]
        action_obj = BlinkAction(move=move, look=look, inspect=bool(inspect))
        obs, reward, done = self._env.step(action_obj)
        return self._encode_observation(obs), reward, done, False, {}

    def render(self):
        state = self._env.state()
        print("Grid:")
        for row in state.full_grid:
            print("".join(row))
        print("agent_pos=", state.agent_pos, "look_center=", state.look_center)

    def _encode_observation(self, obs):
        patch = [CHAR_MAP[cell] for row in obs.patch for cell in row]
        return np.array(
            patch
            + obs.agent_pos
            + obs.look_center
            + [obs.step_count],
            dtype=np.int8,
        )
