from __future__ import annotations

import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple

# 👉  Sole physics source -- imported from professor's reference file
from referencefiles.multibot_cluster_env import MultiBotClusterEnv


class HydroSwarmEnv(MultiBotClusterEnv):
    """
    Decentralised swarm environment for point-A → point-B navigation that
    *re-uses the exact 2-D spinning-bot physics* defined in
    `referencefiles/multibot_cluster_env.py`.

    •  Observation = own state + short-range "sonar" + neighbour snapshot  
    •  Action      = one spinning-frequency value shared by all robots  
    •  Reward      = negative COM-to-target distance  (+ bonus on success)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        n_robots: int = 5,
        dt: float = 0.05,
        episode_seconds: float = 25.0,
        sonar_beams: int = 24,
    ):
        # —— call reference-physics constructor ——————————————— #
        super().__init__(num_bots=n_robots, dt=dt, T=episode_seconds, task="navigate")

        # Target definition (2-D water tank)
        self.start = np.array([-6.0, 0.0])
        self.goal = np.array([6.0, 0.0])
        self.success_threshold = 1.0
        self.max_steps = int(episode_seconds / dt)

        # Replace action space: shared scalar ∈ [-2, 2]
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(self.N,), dtype=np.float32)

        # Observation size: own(6) + sonar + neighbours
        self.sonar_beams = sonar_beams
        own_dim = 6  # pos(2)+vel(2)+goal_rel(2)
        neigh_dim = 4 * (self.N - 1)
        obs_dim = own_dim + sonar_beams + neigh_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Runtime trackers
        self.step_idx = 0
        self.velocities = np.zeros((self.N, 2), dtype=np.float32)

    # --------------------------------------------------------------------- #
    #  Gym API
    # --------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.step_idx = 0
        self._init_formation()
        obs = self._observe()
        return obs.astype(np.float32), {}

    def step(self, action):
        omega = np.full(self.N, np.clip(action[0], -2.0, 2.0))  # shared policy

        old_pos = self.state.reshape(self.N, 2).copy()
        self.state = self._rk4(self.state, omega)               # <- reference physics
        new_pos = self.state.reshape(self.N, 2)
        self.velocities = (new_pos - old_pos) / self.dt

        self.step_idx += 1
        obs = self._observe()
        dist = self._distance_to_goal()
        reward = -dist
        done = dist < self.success_threshold
        truncated = self.step_idx >= self.max_steps

        info = {
            "distance": dist,
            "positions": new_pos.tolist(),      # <- added for visualiser
            "velocities": self.velocities.tolist(),
        }
        return obs.astype(np.float32), reward, done, truncated, info

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _init_formation(self):
        """Hexagonal start-up pattern centred on self.start."""
        positions = np.zeros((self.N, 2))
        positions[0] = self.start
        if self.N > 1:
            angles = np.linspace(0, 2 * np.pi, self.N, endpoint=False)[1:]
            for i, a in enumerate(angles, 1):
                positions[i] = self.start + 1.2 * np.array([np.cos(a), np.sin(a)])
        self.state = positions.flatten()

    def _observe(self) -> np.ndarray:
        pos = self.state.reshape(self.N, 2)
        own = pos[0]
        own_vel = self.velocities[0]
        goal_rel = self.goal - own
        own_state = np.concatenate([own, own_vel, goal_rel])

        # Simple sonar = radial distances to neighbours (padded)
        diffs = pos - own
        dists = np.linalg.norm(diffs, axis=1)
        sonar = np.full(self.sonar_beams, 10.0, dtype=np.float32)
        k = min(self.sonar_beams, self.N - 1)
        sonar[:k] = np.sort(dists[1:])[:k]

        # Neighbour snapshot (rel-pos & rel-vel)
        neigh_info = []
        for j in range(1, self.N):
            neigh_info.extend(np.concatenate([diffs[j], self.velocities[j]]))
        while len(neigh_info) < 4 * (self.N - 1):
            neigh_info.extend([0.0] * 4)

        return np.concatenate([own_state, sonar, neigh_info])

    def _distance_to_goal(self) -> float:
        com = np.mean(self.state.reshape(self.N, 2), axis=0)
        return float(np.linalg.norm(com - self.goal)) 