import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

class EnhancedMultiBotEnv(gym.Env):
    """
    Enhanced environment for multi-bot swarm with:
    - Configurable number of bots (N)
    - Multiple reward functions
    - Support for different tasks
    - Better metrics tracking
    """
    
    def __init__(self, 
                 num_bots: int = 3,
                 dt: float = 0.05,
                 T: float = 10.0,
                 task: str = "formation",
                 reward_type: str = "dense",
                 target_point: Tuple[float, float] = (5.0, 5.0),
                 formation_type: str = "line",
                 **kwargs):
        super().__init__()
        
        # Environment parameters
        self.N = num_bots
        self.dt = dt
        self.T = T
        self.H = int(T / dt)
        self.t = 0
        self.task = task
        self.reward_type = reward_type
        self.target_point = np.array(target_point)
        self.formation_type = formation_type
        
        # Physics parameters
        self.alpha = kwargs.get('alpha', 0.7)
        self.beta = kwargs.get('beta', 0.7)
        self.R0 = kwargs.get('R0', 0.5)
        self.f0 = kwargs.get('f0', 0.05)
        
        # RL spaces - scale with number of bots
        self.action_space = spaces.Box(
            low=-2., high=2., 
            shape=(self.N,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.N * 2,), 
            dtype=np.float32
        )
        
        # Initialize bot positions
        self._init_positions()
        
        # Metrics tracking
        self.episode_metrics = {
            'formation_error': [],
            'velocity_magnitude': [],
            'inter_bot_distances': [],
            'target_distance': [],
            'energy_consumption': []
        }
    
    def _init_positions(self):
        """Initialize bot positions based on formation type"""
        if self.formation_type == "line":
            s = 2.0
            xs = np.linspace(-s, s, self.N)
            ys = np.zeros_like(xs)
        elif self.formation_type == "circle":
            angles = np.linspace(0, 2*np.pi, self.N, endpoint=False)
            radius = 1.5
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
        elif self.formation_type == "grid":
            side = int(np.ceil(np.sqrt(self.N)))
            positions = []
            for i in range(self.N):
                row = i // side
                col = i % side
                positions.append([col - side/2, row - side/2])
            xs, ys = zip(*positions)
            xs, ys = np.array(xs), np.array(ys)
        else:  # random
            xs = np.random.uniform(-2, 2, self.N)
            ys = np.random.uniform(-2, 2, self.N)
            
        self.X0 = np.vstack([xs, ys]).T.reshape(-1)
        self.state = self.X0.copy()
    
    def _forces(self, X: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Compute forces between bots"""
        pos = X.reshape(self.N, 2)
        forces = np.zeros_like(pos)
        
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                rij = pos[j] - pos[i]
                dist = np.linalg.norm(rij) + 1e-6
                r_hat = rij / dist
                
                # Attractive/repulsive force
                F_r = self.alpha * omega[i] / dist - self.beta / ((dist - self.R0)**6 + 1e-3)
                # Tangential force
                F_t = self.f0 * omega[i] * np.array([-r_hat[1], r_hat[0]])
                
                forces[i] += F_r * r_hat + F_t
        
        return forces.reshape(-1)
    
    def _rk4(self, X: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """4th order Runge-Kutta integration"""
        k1 = self._forces(X, omega)
        k2 = self._forces(X + 0.5 * self.dt * k1, omega)
        k3 = self._forces(X + 0.5 * self.dt * k2, omega)
        k4 = self._forces(X + self.dt * k3, omega)
        return X + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    def _compute_reward(self, X: np.ndarray, action: np.ndarray) -> float:
        """Compute reward based on task and reward type"""
        P = X.reshape(self.N, 2)
        
        if self.task == "formation":
            return self._formation_reward(P, action)
        elif self.task == "translation":
            return self._translation_reward(P, action)
        elif self.task == "coverage":
            return self._coverage_reward(P, action)
        elif self.task == "flocking":
            return self._flocking_reward(P, action)
        else:
            return 0.0
    
    def _formation_reward(self, P: np.ndarray, action: np.ndarray) -> float:
        """Reward for maintaining formation"""
        if self.formation_type == "line":
            # Minimize variance in y-direction for line formation
            line_error = np.var(P[:, 1])
            if self.reward_type == "dense":
                return -line_error - 0.01 * np.sum(action**2)
            else:  # sparse
                return -1.0 if line_error > 0.1 else 10.0
                
        elif self.formation_type == "circle":
            # Maintain circular formation
            center = P.mean(axis=0)
            distances = np.linalg.norm(P - center, axis=1)
            circle_error = np.var(distances)
            if self.reward_type == "dense":
                return -circle_error - 0.01 * np.sum(action**2)
            else:
                return -1.0 if circle_error > 0.1 else 10.0
        
        return 0.0
    
    def _translation_reward(self, P: np.ndarray, action: np.ndarray) -> float:
        """Reward for moving to target while maintaining formation"""
        com = P.mean(axis=0)
        target_dist = np.linalg.norm(com - self.target_point)
        
        # Formation maintenance (line formation)
        formation_error = np.var(P[:, 1]) if self.formation_type == "line" else 0
        
        if self.reward_type == "dense":
            reward = -target_dist - formation_error - 0.01 * np.sum(action**2)
        else:  # sparse
            if target_dist < 0.5 and formation_error < 0.1:
                reward = 100.0
            elif target_dist < 1.0:
                reward = 10.0
            else:
                reward = -1.0
                
        return reward
    
    def _coverage_reward(self, P: np.ndarray, action: np.ndarray) -> float:
        """Reward for area coverage"""
        # Maximize area covered by bots
        if self.N < 3:
            return 0.0
            
        # Convex hull area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(P)
            area = hull.volume  # In 2D, volume is area
            if self.reward_type == "dense":
                return area - 0.01 * np.sum(action**2)
            else:
                return 10.0 if area > 5.0 else -1.0
        except:
            return -10.0
    
    def _flocking_reward(self, P: np.ndarray, action: np.ndarray) -> float:
        """Reward for flocking behavior (cohesion + alignment + separation)"""
        if self.N < 2:
            return 0.0
        
        # Cohesion: move towards center of mass
        com = P.mean(axis=0)
        cohesion = -np.mean(np.linalg.norm(P - com, axis=1))
        
        # Separation: avoid crowding
        min_dist = float('inf')
        for i in range(self.N):
            for j in range(i+1, self.N):
                dist = np.linalg.norm(P[i] - P[j])
                min_dist = min(min_dist, dist)
        
        separation = min_dist if min_dist > 0.5 else -10.0
        
        if self.reward_type == "dense":
            return cohesion + separation - 0.01 * np.sum(action**2)
        else:
            good_formation = (min_dist > 0.3 and np.mean(np.linalg.norm(P - com, axis=1)) < 2.0)
            return 10.0 if good_formation else -1.0
    
    def _update_metrics(self, X: np.ndarray, action: np.ndarray):
        """Update episode metrics"""
        P = X.reshape(self.N, 2)
        
        # Formation error
        if self.formation_type == "line":
            formation_error = np.var(P[:, 1])
        else:
            formation_error = 0.0
        
        # Inter-bot distances
        distances = []
        for i in range(self.N):
            for j in range(i+1, self.N):
                distances.append(np.linalg.norm(P[i] - P[j]))
        
        self.episode_metrics['formation_error'].append(formation_error)
        self.episode_metrics['inter_bot_distances'].append(np.mean(distances))
        self.episode_metrics['target_distance'].append(
            np.linalg.norm(P.mean(axis=0) - self.target_point)
        )
        self.episode_metrics['energy_consumption'].append(np.sum(action**2))
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._init_positions()
        self.state = self.X0 + np.random.normal(scale=0.1, size=self.X0.shape)
        self.t = 0
        
        # Reset metrics
        for key in self.episode_metrics:
            self.episode_metrics[key] = []
            
        info = {'episode_metrics': self.episode_metrics}
        return self.state.astype(np.float32), info
    
    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Update state
        next_state = self._rk4(self.state, action)
        reward = self._compute_reward(next_state, action)
        
        # Update metrics
        self._update_metrics(next_state, action)
        
        self.state = next_state
        self.t += 1
        
        terminated = False
        truncated = self.t >= self.H
        
        info = {
            'episode_metrics': self.episode_metrics,
            'current_formation_error': self.episode_metrics['formation_error'][-1] if self.episode_metrics['formation_error'] else 0,
            'current_target_distance': self.episode_metrics['target_distance'][-1] if self.episode_metrics['target_distance'] else 0
        }
        
        return next_state.astype(np.float32), reward, terminated, truncated, info
    
    def render(self):
        """Optional visualization"""
        pass 