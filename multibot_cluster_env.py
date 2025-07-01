import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiBotClusterEnv(gym.Env):
    """
    Gymnasium-compatible environment for controlling a cluster of self-propelled bots
    with spinning frequencies. Supports multiple tasks: line formation, translation, shape.
    """
    metadata = {
        "render_modes": ["human"],
        "render_fps": 10
    }

    def __init__(self, num_bots=3, dt=0.05, T=10.0, task="line"):
        super().__init__()
        self.N = num_bots
        self.dt = dt
        self.T = T
        self.H = int(T / dt)
        self.t = 0
        self.task = task

        self.alpha = 0.7
        self.beta = 0.7
        self.R0 = 0.5
        self.f0 = 0.05

        # RL spaces
        self.action_space = spaces.Box(low=-2., high=2., shape=(self.N,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.N * 2,), dtype=np.float32)

        # Initial configuration: horizontal line
        s = 2.0
        xs = np.linspace(-s, s, self.N)
        ys = np.zeros_like(xs)
        self.X0 = np.vstack([xs, ys]).T.reshape(-1)
        self.state = self.X0.copy()

    def _forces(self, X, omega):
        pos = X.reshape(self.N, 2)
        forces = np.zeros_like(pos)
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                rij = pos[j] - pos[i]
                dist = np.linalg.norm(rij) + 1e-6
                r_hat = rij / dist
                F_r = self.alpha * omega[i] / dist - self.beta / ((dist - self.R0)**6 + 1e-3)
                F_t = self.f0 * omega[i] * np.array([-r_hat[1], r_hat[0]])
                forces[i] += F_r * r_hat + F_t
        return forces.reshape(-1)

    def _rk4(self, X, omega):
        k1 = self._forces(X, omega)
        k2 = self._forces(X + 0.5 * self.dt * k1, omega)
        k3 = self._forces(X + 0.5 * self.dt * k2, omega)
        k4 = self._forces(X + self.dt * k3, omega)
        return X + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _reward(self, X):
        P = X.reshape(self.N, 2)
        if self.task == "line":
            return -np.var(P[:, 1])
        elif self.task == "translate":
            target = np.array([5., 5.])
            com = P.mean(axis=0)
            return -np.linalg.norm(com - target)**2
        elif self.task == "shape":
            dist_sum = np.sum([np.linalg.norm(P[i] - P[j])
                               for i in range(self.N) for j in range(i+1, self.N)])
            return dist_sum / (self.N * (self.N - 1) / 2)
        return 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.X0 + np.random.normal(scale=0.1, size=self.X0.shape)
        self.t = 0
        info = {}
        return self.state.astype(np.float32), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        nxt = self._rk4(self.state, action)
        reward = self._reward(nxt)
        self.state = nxt
        self.t += 1
        terminated = False  # could use task completion here
        truncated = self.t >= self.H
        info = {}
        return nxt.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        pass  # Optional: live visualization
