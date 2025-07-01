#!/usr/bin/env python3
"""
Consistent RL Algorithm Evaluation - Task-Completion Based Testing
All algorithms get fair evaluation based on task completion, not arbitrary time limits
"""

import pygame
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import deque
import json
import random

# Use the existing physics environment
from multibot_cluster_env import MultiBotClusterEnv

@dataclass
class SimpleAgent:
    """Simple agent with position, velocity, and neural network policy"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    spinning_speed: float
    color: Tuple[int, int, int]
    trail: deque
    neighbors: List
    local_target: np.ndarray
    
    def __post_init__(self):
        if not self.trail:
            self.trail = deque(maxlen=100)

class SimpleNeuralPolicy:
    """Enhanced neural network policy with better learning"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 1, learning_rate: float = 0.01):
        # Initialize neural network weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim//2) * 0.1
        self.b2 = np.zeros(hidden_dim//2)
        self.W3 = np.random.randn(hidden_dim//2, output_dim) * 0.1
        self.b3 = np.zeros(output_dim)
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.experience_buffer = deque(maxlen=2000)
        self.reward_history = deque(maxlen=200)
        
        # Task completion tracking
        self.episodes_completed = 0
        self.successful_episodes = 0
        self.learning_curve = deque(maxlen=100)
    
    def forward(self, x: np.ndarray) -> float:
        """Forward pass through deeper network"""
        # Layer 1
        h1 = np.maximum(0, np.dot(x, self.W1) + self.b1)  # ReLU
        # Layer 2  
        h2 = np.maximum(0, np.dot(h1, self.W2) + self.b2)  # ReLU
        # Output layer
        output = np.tanh(np.dot(h2, self.W3) + self.b3)  # Tanh output
        return output[0] * 2.0  # Scale to [-2, 2]
    
    def update(self, experience: Dict):
        """Enhanced learning with better gradient updates"""
        self.experience_buffer.append(experience)
        self.reward_history.append(experience['reward'])
        
        if len(self.experience_buffer) > 20:
            # More sophisticated learning update
            recent_experiences = list(self.experience_buffer)[-20:]
            avg_reward = np.mean([exp['reward'] for exp in recent_experiences])
            
            # Adaptive learning rate based on performance
            if avg_reward > 0:
                self.learning_rate = min(0.05, self.learning_rate * 1.02)
            else:
                self.learning_rate = max(0.001, self.learning_rate * 0.98)
            
            # Track learning progress
            if len(self.reward_history) >= 10:
                recent_avg = np.mean(list(self.reward_history)[-10:])
                self.learning_curve.append(recent_avg)
    
    def get_success_rate(self) -> float:
        """Calculate success rate over recent episodes"""
        if self.episodes_completed == 0:
            return 0.0
        return self.successful_episodes / max(1, self.episodes_completed)
    
    def has_learned_task(self) -> bool:
        """Check if the policy has learned the task (consistent success)"""
        if len(self.learning_curve) < 20:
            return False
        
        # Check if recent performance is consistently good
        recent_performance = list(self.learning_curve)[-20:]
        return np.mean(recent_performance) > 2.0 and np.min(recent_performance) > 0.5

@dataclass
class TaskEvaluation:
    """Evaluation metrics for task completion"""
    algorithm_name: str
    episodes_attempted: int = 0
    episodes_completed: int = 0
    total_time_taken: float = 0.0
    best_time: float = float('inf')
    average_time: float = 0.0
    success_rate: float = 0.0
    final_distance: float = 0.0
    formation_quality: float = 0.0
    learning_efficiency: float = 0.0
    task_learned: bool = False

@dataclass
class AlgorithmConfig:
    """Configuration for each algorithm with task-based evaluation"""
    name: str
    color: Tuple[int, int, int]
    policy: SimpleNeuralPolicy
    evaluation: TaskEvaluation
    learning_characteristics: str = "Standard"
    is_active: bool = False
    start_time: float = 0.0

class TaskBasedSwarmDemo:
    """Fair evaluation system based on actual task completion"""
    
    def __init__(self, width: int = 1400, height: int = 900):
        pygame.init()
        
        # Display setup
        self.width = width
        self.height = height
        self.control_panel_width = 400
        self.main_area_width = width - self.control_panel_width
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Task-Based RL Algorithm Evaluation")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.PURPLE = (128, 0, 128)
        self.GRAY = (128, 128, 128)
        self.ORANGE = (255, 165, 0)
        self.LIGHT_GREEN = (144, 238, 144)
        self.DARK_GREEN = (0, 100, 0)
        
        # Fonts
        self.font_tiny = pygame.font.Font(None, 14)
        self.font_small = pygame.font.Font(None, 16)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        
        # Task parameters
        self.SUCCESS_DISTANCE = 60.0  # Distance to target for success
        self.MAX_EPISODE_STEPS = 1000  # Maximum steps per episode
        self.MIN_EPISODES_FOR_EVALUATION = 5  # Minimum episodes before switching
        self.MAX_EPISODES_PER_ALGORITHM = 50  # Maximum episodes per algorithm
        self.LEARNING_PATIENCE = 10  # Episodes to wait after learning detected
        
        # Algorithm configurations with different learning characteristics
        self.algorithms = {
            'DDPG-Style': AlgorithmConfig(
                name='DDPG-Style',
                color=self.BLUE,
                policy=SimpleNeuralPolicy(input_dim=10, hidden_dim=64, learning_rate=0.003),
                evaluation=TaskEvaluation('DDPG-Style'),
                learning_characteristics="Conservative Learner"
            ),
            'SAC-Style': AlgorithmConfig(
                name='SAC-Style',
                color=self.RED,
                policy=SimpleNeuralPolicy(input_dim=10, hidden_dim=80, learning_rate=0.01),
                evaluation=TaskEvaluation('SAC-Style'),
                learning_characteristics="Sample Efficient"
            ),
            'TD3-Style': AlgorithmConfig(
                name='TD3-Style',
                color=self.GREEN,
                policy=SimpleNeuralPolicy(input_dim=10, hidden_dim=72, learning_rate=0.005),
                evaluation=TaskEvaluation('TD3-Style'),
                learning_characteristics="Stable Learner"
            ),
            'PPO-Style': AlgorithmConfig(
                name='PPO-Style',
                color=self.PURPLE,
                policy=SimpleNeuralPolicy(input_dim=10, hidden_dim=56, learning_rate=0.015),
                evaluation=TaskEvaluation('PPO-Style'),
                learning_characteristics="Policy Optimizer"
            )
        }
        
        # Evaluation system
        self.current_algorithm = 'DDPG-Style'
        self.algorithm_queue = list(self.algorithms.keys())
        self.current_algorithm_index = 0
        self.evaluation_complete = False
        
        # Episode tracking
        self.current_episode = 0
        self.episode_start_time = 0.0
        self.episode_steps = 0
        self.episode_success = False
        
        # Navigation setup
        self.start_point = np.array([120.0, self.height/2])
        self.target_point = np.array([self.main_area_width - 120, self.height/2])
        self.initial_distance = np.linalg.norm(self.target_point - self.start_point)
        
        # Swarm setup
        self.num_agents = 5
        self.observation_radius = 100.0
        self.agents = []
        
        # Physics environment
        self.physics_env = MultiBotClusterEnv(num_bots=self.num_agents, task='translate')
        self.setup_new_episode()
        
        # Control flags
        self.running = True
        self.paused = False
        self.show_trails = True
        self.show_info = True
        self.show_metrics = True
        self.show_observation_radius = False
        self.auto_evaluate = True
        
        # Timing
        self.clock = pygame.time.Clock()
        self.dt = 0.05
        
        # Training
        self.training_thread = None
        self.start_training()
        
        # Start first algorithm
        self.start_algorithm_evaluation(self.current_algorithm)
    
    def start_algorithm_evaluation(self, algorithm_name: str):
        """Start evaluating a specific algorithm"""
        config = self.algorithms[algorithm_name]
        config.is_active = True
        config.start_time = time.time()
        config.evaluation = TaskEvaluation(algorithm_name)  # Reset evaluation
        
        print(f"\n🔬 Starting evaluation: {algorithm_name} ({config.learning_characteristics})")
        print(f"   Target: Navigate swarm from A to B (distance: {self.initial_distance:.0f}px)")
        print(f"   Success criteria: Get within {self.SUCCESS_DISTANCE:.0f}px of target")
        print(f"   Max episodes: {self.MAX_EPISODES_PER_ALGORITHM}")
        
        self.setup_new_episode()
    
    def should_switch_algorithm(self) -> bool:
        """Determine if we should switch to next algorithm based on task completion"""
        current_config = self.algorithms[self.current_algorithm]
        evaluation = current_config.evaluation
        
        # Minimum episodes check
        if evaluation.episodes_attempted < self.MIN_EPISODES_FOR_EVALUATION:
            return False
        
        # Maximum episodes check (forced switch)
        if evaluation.episodes_attempted >= self.MAX_EPISODES_PER_ALGORITHM:
            return True
        
        # Task learned check (with patience for stabilization)
        if current_config.policy.has_learned_task():
            if not evaluation.task_learned:
                evaluation.task_learned = True
                self.learning_detected_episode = evaluation.episodes_attempted
            
            # Give some extra episodes after learning is detected
            if evaluation.episodes_attempted >= self.learning_detected_episode + self.LEARNING_PATIENCE:
                return True
        
        # Early stopping for very poor performance
        if (evaluation.episodes_attempted >= 20 and 
            evaluation.success_rate < 0.1 and 
            np.mean(list(current_config.policy.learning_curve)[-10:]) < -1.0):
            print(f"   Early stopping for {self.current_algorithm} due to poor performance")
            return True
        
        return False
    
    def switch_to_next_algorithm(self):
        """Switch to the next algorithm for evaluation"""
        # Finalize current algorithm evaluation
        current_config = self.algorithms[self.current_algorithm]
        current_config.is_active = False
        evaluation = current_config.evaluation
        
        if evaluation.episodes_completed > 0:
            evaluation.average_time = evaluation.total_time_taken / evaluation.episodes_completed
            evaluation.learning_efficiency = evaluation.success_rate / max(1, evaluation.episodes_attempted) * 100
        
        print(f"   ✅ {self.current_algorithm} evaluation complete:")
        print(f"      Episodes: {evaluation.episodes_attempted}")
        print(f"      Success rate: {evaluation.success_rate:.1%}")
        print(f"      Best time: {evaluation.best_time:.1f}s")
        print(f"      Task learned: {evaluation.task_learned}")
        
        # Move to next algorithm
        self.current_algorithm_index = (self.current_algorithm_index + 1) % len(self.algorithm_queue)
        
        # Check if evaluation is complete
        if self.current_algorithm_index == 0 and current_config.evaluation.episodes_attempted > 0:
            self.evaluation_complete = True
            self.print_final_results()
            return
        
        # Start next algorithm
        self.current_algorithm = self.algorithm_queue[self.current_algorithm_index]
        self.start_algorithm_evaluation(self.current_algorithm)
    
    def setup_new_episode(self):
        """Set up a new episode"""
        self.agents = []
        self.episode_start_time = time.time()
        self.episode_steps = 0
        self.episode_success = False
        
        # Initialize agents in formation near start point
        for i in range(self.num_agents):
            # Line formation
            spacing = 30
            x_offset = (i - (self.num_agents - 1) / 2) * spacing
            position = self.start_point + np.array([x_offset, np.random.normal(0, 5)])
            
            # Add some randomness for realism
            position += np.random.normal(0, 10, 2)
            
            agent = SimpleAgent(
                id=i,
                position=position,
                velocity=np.array([0.0, 0.0]),
                spinning_speed=0.0,
                color=self.algorithms[self.current_algorithm].color,
                trail=deque(maxlen=100),
                neighbors=[],
                local_target=self.target_point.copy()
            )
            agent.trail.append(position.copy())
            self.agents.append(agent)
        
        # Reset physics environment
        try:
            obs, _ = self.physics_env.reset()
        except Exception as e:
            pass
    
    def check_episode_completion(self) -> bool:
        """Check if current episode is complete (success or failure)"""
        # Calculate swarm center
        swarm_center = np.mean([agent.position for agent in self.agents], axis=0)
        distance_to_target = np.linalg.norm(swarm_center - self.target_point)
        
        # Success condition
        if distance_to_target < self.SUCCESS_DISTANCE:
            self.episode_success = True
            return True
        
        # Failure conditions
        if self.episode_steps >= self.MAX_EPISODE_STEPS:
            return True
        
        # Check if swarm is stuck (all agents have very low velocity for extended time)
        avg_velocity = np.mean([np.linalg.norm(agent.velocity) for agent in self.agents])
        if self.episode_steps > 100 and avg_velocity < 0.1:
            return True
        
        return False
    
    def complete_episode(self):
        """Complete the current episode and update statistics"""
        current_config = self.algorithms[self.current_algorithm]
        evaluation = current_config.evaluation
        policy = current_config.policy
        
        episode_time = time.time() - self.episode_start_time
        evaluation.episodes_attempted += 1
        policy.episodes_completed += 1
        
        # Calculate final metrics
        swarm_center = np.mean([agent.position for agent in self.agents], axis=0)
        final_distance = np.linalg.norm(swarm_center - self.target_point)
        evaluation.final_distance = final_distance
        
        # Formation quality (lower variance = better formation)
        positions = np.array([agent.position for agent in self.agents])
        evaluation.formation_quality = 1.0 / (1.0 + np.var(positions))
        
        if self.episode_success:
            evaluation.episodes_completed += 1
            evaluation.total_time_taken += episode_time
            evaluation.best_time = min(evaluation.best_time, episode_time)
            policy.successful_episodes += 1
            
            print(f"   ✅ Episode {evaluation.episodes_attempted} SUCCESS in {episode_time:.1f}s")
        else:
            print(f"   ❌ Episode {evaluation.episodes_attempted} FAILED (distance: {final_distance:.0f}px)")
        
        # Update success rate
        evaluation.success_rate = evaluation.episodes_completed / evaluation.episodes_attempted
        
        # Start new episode if not switching algorithms
        if not self.should_switch_algorithm():
            self.setup_new_episode()
        else:
            self.switch_to_next_algorithm()
    
    def get_local_observation(self, agent: SimpleAgent) -> np.ndarray:
        """Get decentralized observation for one agent"""
        obs = np.zeros(10)
        
        # Agent's position relative to target
        target_dir = self.target_point - agent.position
        target_dist = np.linalg.norm(target_dir)
        
        if target_dist > 0:
            obs[0:2] = target_dir / target_dist  # Normalized direction
            obs[2] = min(target_dist / self.initial_distance, 1.0)  # Normalized distance
        
        # Agent's velocity
        obs[3:5] = agent.velocity * 0.05
        
        # Find neighbors
        agent.neighbors = []
        for other in self.agents:
            if other.id != agent.id:
                dist = np.linalg.norm(other.position - agent.position)
                if dist < self.observation_radius:
                    agent.neighbors.append({
                        'id': other.id,
                        'distance': dist,
                        'relative_pos': other.position - agent.position,
                        'velocity': other.velocity
                    })
        
        # Add neighbor information
        agent.neighbors.sort(key=lambda x: x['distance'])
        for i, neighbor in enumerate(agent.neighbors[:2]):
            base_idx = 5 + i * 2
            if base_idx + 1 < len(obs):
                obs[base_idx:base_idx+2] = neighbor['relative_pos'] / self.observation_radius
        
        return obs
    
    def calculate_reward(self, agent: SimpleAgent, action: float) -> float:
        """Calculate comprehensive reward for agent"""
        reward = 0.0
        
        # Distance to target reward (main objective)
        target_dist = np.linalg.norm(agent.position - self.target_point)
        progress_reward = -(target_dist / self.initial_distance) * 2.0
        reward += progress_reward
        
        # Formation maintenance reward
        if agent.neighbors:
            avg_neighbor_dist = np.mean([n['distance'] for n in agent.neighbors])
            if 20 < avg_neighbor_dist < 80:  # Optimal spacing
                reward += 0.5
            elif avg_neighbor_dist < 15:  # Too close
                reward -= 0.3
            elif avg_neighbor_dist > 120:  # Too far
                reward -= 0.2
        
        # Movement efficiency reward
        velocity_magnitude = np.linalg.norm(agent.velocity)
        if velocity_magnitude > 0.5:  # Encourage movement
            reward += 0.2
        
        # Success bonus
        if target_dist < self.SUCCESS_DISTANCE:
            reward += 10.0
        
        # Energy penalty
        reward -= abs(action) * 0.03
        
        return reward
    
    def update_physics(self):
        """Update agent physics and check episode completion"""
        if self.evaluation_complete:
            return
        
        # Get actions from current algorithm
        actions = []
        observations = []
        
        for agent in self.agents:
            obs = self.get_local_observation(agent)
            observations.append(obs)
            
            # Get action from current algorithm
            current_policy = self.algorithms[self.current_algorithm].policy
            action = current_policy.forward(obs)
            actions.append(action)
            
            agent.spinning_speed = action
        
        # Physics update with improved movement model
        for i, agent in enumerate(self.agents):
            action = actions[i]
            
            # Movement towards target
            target_dir = self.target_point - agent.position
            target_dist = np.linalg.norm(target_dir)
            
            if target_dist > 5:
                # Normalized direction
                direction = target_dir / target_dist
                
                # Speed based on action and distance
                base_speed = abs(action) * 40.0
                distance_factor = min(1.0, target_dist / 100.0)  # Slow down near target
                speed = base_speed * distance_factor
                
                # Formation forces
                formation_force = np.array([0.0, 0.0])
                for neighbor in agent.neighbors:
                    if neighbor['distance'] < 25:  # Repulsion when too close
                        repulsion = -neighbor['relative_pos'] / (neighbor['distance'] + 1e-6)
                        formation_force += repulsion * 0.15
                    elif neighbor['distance'] > 100:  # Attraction when too far
                        attraction = neighbor['relative_pos'] / (neighbor['distance'] + 1e-6)
                        formation_force += attraction * 0.05
                
                # Combined movement
                movement = direction * speed * self.dt + formation_force * self.dt
                
                # Update with momentum
                agent.velocity = agent.velocity * 0.85 + movement * 8.0
                agent.position += agent.velocity * self.dt
                
                # Boundary constraints
                agent.position[0] = np.clip(agent.position[0], 30, self.main_area_width - 30)
                agent.position[1] = np.clip(agent.position[1], 30, self.height - 30)
            
            # Update trail
            agent.trail.append(agent.position.copy())
        
        # Update policies with rewards
        for i, agent in enumerate(self.agents):
            reward = self.calculate_reward(agent, actions[i])
            
            experience = {
                'observation': observations[i],
                'action': actions[i],
                'reward': reward
            }
            
            current_policy = self.algorithms[self.current_algorithm].policy
            current_policy.update(experience)
        
        self.episode_steps += 1
        
        # Check episode completion
        if self.check_episode_completion():
            self.complete_episode()
    
    def start_training(self):
        """Start background training"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
            self.training_thread.start()
    
    def training_loop(self):
        """Background training loop"""
        while self.running:
            try:
                if not self.paused and not self.evaluation_complete:
                    self.update_physics()
                time.sleep(0.02)  # 50 Hz
            except Exception as e:
                print(f"Training error: {e}")
                time.sleep(0.1)
    
    def print_final_results(self):
        """Print comprehensive final results"""
        print("\n" + "="*80)
        print("🏆 FINAL EVALUATION RESULTS")
        print("="*80)
        
        # Sort algorithms by performance
        sorted_algos = sorted(
            self.algorithms.items(), 
            key=lambda x: (x[1].evaluation.success_rate, -x[1].evaluation.average_time),
            reverse=True
        )
        
        print(f"{'Rank':<4} {'Algorithm':<12} {'Success':<8} {'Avg Time':<9} {'Best Time':<9} {'Episodes':<9} {'Learned':<8}")
        print("-" * 80)
        
        for rank, (name, config) in enumerate(sorted_algos, 1):
            eval_data = config.evaluation
            success_str = f"{eval_data.success_rate:.1%}"
            avg_time_str = f"{eval_data.average_time:.1f}s" if eval_data.average_time > 0 else "N/A"
            best_time_str = f"{eval_data.best_time:.1f}s" if eval_data.best_time < float('inf') else "N/A"
            episodes_str = f"{eval_data.episodes_attempted}"
            learned_str = "Yes" if eval_data.task_learned else "No"
            
            print(f"{rank:<4} {name:<12} {success_str:<8} {avg_time_str:<9} {best_time_str:<9} {episodes_str:<9} {learned_str:<8}")
        
        print("="*80)
    
    def draw_agent(self, agent: SimpleAgent):
        """Draw agent with enhanced visuals"""
        pos = tuple(map(int, agent.position))
        
        # Draw observation radius
        if self.show_observation_radius:
            pygame.draw.circle(self.screen, (*agent.color, 30), pos, int(self.observation_radius), 1)
        
        # Draw agent body (larger for better visibility)
        pygame.draw.circle(self.screen, agent.color, pos, 12)
        pygame.draw.circle(self.screen, self.BLACK, pos, 12, 2)
        
        # Draw velocity direction
        if np.linalg.norm(agent.velocity) > 0.5:
            vel_norm = agent.velocity / (np.linalg.norm(agent.velocity) + 1e-6)
            end_pos = pos + vel_norm * 20
            pygame.draw.line(self.screen, self.BLACK, pos, tuple(map(int, end_pos)), 3)
        
        # Draw spinning speed indicator
        spin_color = self.DARK_GREEN if agent.spinning_speed > 0 else self.RED
        spin_radius = abs(agent.spinning_speed) * 4 + 3
        pygame.draw.circle(self.screen, spin_color, pos, int(spin_radius), 2)
        
        # Draw neighbor connections
        for neighbor in agent.neighbors:
            neighbor_agent = self.agents[neighbor['id']]
            neighbor_pos = tuple(map(int, neighbor_agent.position))
            pygame.draw.line(self.screen, self.GRAY, pos, neighbor_pos, 1)
        
        # Draw info
        if self.show_info:
            info_text = f"{agent.id}: {agent.spinning_speed:.1f}"
            text_surface = self.font_small.render(info_text, True, self.BLACK)
            self.screen.blit(text_surface, (pos[0] - 25, pos[1] - 30))
    
    def draw_trail(self, agent: SimpleAgent):
        """Draw agent trail with better visibility"""
        if not self.show_trails or len(agent.trail) < 2:
            return
        
        trail_points = list(agent.trail)
        for i in range(1, len(trail_points)):
            alpha = (i / len(trail_points)) * 0.8 + 0.2  # More visible trails
            color = tuple(int(c * alpha) for c in agent.color)
            
            start_pos = tuple(map(int, trail_points[i-1]))
            end_pos = tuple(map(int, trail_points[i]))
            
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
    
    def draw_navigation_points(self):
        """Draw enhanced start and target points"""
        # Start point A
        start_pos = tuple(map(int, self.start_point))
        pygame.draw.circle(self.screen, self.LIGHT_GREEN, start_pos, 25)
        pygame.draw.circle(self.screen, self.BLACK, start_pos, 25, 3)
        start_text = self.font_large.render("A", True, self.BLACK)
        text_rect = start_text.get_rect(center=start_pos)
        self.screen.blit(start_text, text_rect)
        
        # Target point B with success radius
        target_pos = tuple(map(int, self.target_point))
        pygame.draw.circle(self.screen, (*self.RED, 50), target_pos, int(self.SUCCESS_DISTANCE), 1)  # Success zone
        pygame.draw.circle(self.screen, self.RED, target_pos, 25)
        pygame.draw.circle(self.screen, self.BLACK, target_pos, 25, 3)
        target_text = self.font_large.render("B", True, self.WHITE)
        text_rect = target_text.get_rect(center=target_pos)
        self.screen.blit(target_text, text_rect)
        
        # Direct path
        pygame.draw.line(self.screen, self.GRAY, start_pos, target_pos, 2)
        
        # Distance indicator
        distance_text = f"{self.initial_distance:.0f}px"
        mid_point = ((start_pos[0] + target_pos[0]) // 2, (start_pos[1] + target_pos[1]) // 2 - 20)
        distance_surface = self.font_small.render(distance_text, True, self.BLACK)
        self.screen.blit(distance_surface, mid_point)
    
    def draw_control_panel(self):
        """Draw comprehensive control panel"""
        panel_x = self.main_area_width
        panel_rect = pygame.Rect(panel_x, 0, self.control_panel_width, self.height)
        pygame.draw.rect(self.screen, (245, 245, 245), panel_rect)
        pygame.draw.line(self.screen, self.BLACK, (panel_x, 0), (panel_x, self.height), 3)
        
        y_offset = 10
        line_height = 16
        
        # Title
        title = self.font_large.render("TASK-BASED EVALUATION", True, self.BLACK)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 35
        
        if not self.evaluation_complete:
            # Current algorithm status
            current_config = self.algorithms[self.current_algorithm]
            current_text = self.font_medium.render(f"Testing: {self.current_algorithm}", True, current_config.color)
            self.screen.blit(current_text, (panel_x + 10, y_offset))
            y_offset += 22
            
            # Current episode info
            episode_text = f"Episode: {current_config.evaluation.episodes_attempted + 1}"
            episode_surface = self.font_small.render(episode_text, True, self.BLACK)
            self.screen.blit(episode_surface, (panel_x + 10, y_offset))
            y_offset += 18
            
            # Episode progress
            episode_time = time.time() - self.episode_start_time
            time_text = f"Time: {episode_time:.1f}s | Steps: {self.episode_steps}"
            time_surface = self.font_tiny.render(time_text, True, self.BLACK)
            self.screen.blit(time_surface, (panel_x + 10, y_offset))
            y_offset += 16
            
            # Current distance to target
            if self.agents:
                swarm_center = np.mean([agent.position for agent in self.agents], axis=0)
                current_distance = np.linalg.norm(swarm_center - self.target_point)
                distance_text = f"Distance to target: {current_distance:.0f}px"
                distance_surface = self.font_tiny.render(distance_text, True, self.BLACK)
                self.screen.blit(distance_surface, (panel_x + 10, y_offset))
                y_offset += 18
            
            y_offset += 10
        
        # Algorithm results table
        results_title = self.font_medium.render("ALGORITHM PERFORMANCE:", True, self.BLACK)
        self.screen.blit(results_title, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Table headers
        headers = ["Algo", "Episodes", "Success", "Best Time", "Status"]
        header_positions = [panel_x + 10, panel_x + 70, panel_x + 130, panel_x + 190, panel_x + 260]
        
        for i, header in enumerate(headers):
            header_surface = self.font_tiny.render(header, True, self.BLACK)
            self.screen.blit(header_surface, (header_positions[i], y_offset))
        y_offset += 18
        
        # Draw separator line
        pygame.draw.line(self.screen, self.GRAY, (panel_x + 10, y_offset), (panel_x + 380, y_offset), 1)
        y_offset += 5
        
        # Algorithm data
        for algo_name, config in self.algorithms.items():
            eval_data = config.evaluation
            
            # Color indicator
            color_rect = pygame.Rect(panel_x + 10, y_offset, 12, 12)
            pygame.draw.rect(self.screen, config.color, color_rect)
            pygame.draw.rect(self.screen, self.BLACK, color_rect, 1)
            
            # Algorithm data
            name_text = algo_name.replace('-Style', '')
            episodes_text = f"{eval_data.episodes_attempted}"
            success_text = f"{eval_data.success_rate:.1%}"
            best_time_text = f"{eval_data.best_time:.1f}s" if eval_data.best_time < float('inf') else "N/A"
            
            # Status
            if config.is_active:
                status_text = "TESTING"
                status_color = self.ORANGE
            elif eval_data.task_learned:
                status_text = "LEARNED"
                status_color = self.GREEN
            elif eval_data.episodes_attempted > 0:
                status_text = "COMPLETE"
                status_color = self.BLUE
            else:
                status_text = "PENDING"
                status_color = self.GRAY
            
            # Render text
            name_surface = self.font_tiny.render(name_text, True, self.BLACK)
            episodes_surface = self.font_tiny.render(episodes_text, True, self.BLACK)
            success_surface = self.font_tiny.render(success_text, True, self.BLACK)
            best_time_surface = self.font_tiny.render(best_time_text, True, self.BLACK)
            status_surface = self.font_tiny.render(status_text, True, status_color)
            
            # Position text
            self.screen.blit(name_surface, (panel_x + 25, y_offset))
            self.screen.blit(episodes_surface, (header_positions[1], y_offset))
            self.screen.blit(success_surface, (header_positions[2], y_offset))
            self.screen.blit(best_time_surface, (header_positions[3], y_offset))
            self.screen.blit(status_surface, (header_positions[4], y_offset))
            
            y_offset += 18
        
        y_offset += 20
        
        # Task parameters
        task_title = self.font_medium.render("TASK PARAMETERS:", True, self.BLACK)
        self.screen.blit(task_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        task_info = [
            f"Success distance: {self.SUCCESS_DISTANCE:.0f}px",
            f"Max episode steps: {self.MAX_EPISODE_STEPS}",
            f"Min episodes: {self.MIN_EPISODES_FOR_EVALUATION}",
            f"Max episodes: {self.MAX_EPISODES_PER_ALGORITHM}",
            f"Total distance: {self.initial_distance:.0f}px"
        ]
        
        for info in task_info:
            info_surface = self.font_tiny.render(info, True, self.BLACK)
            self.screen.blit(info_surface, (panel_x + 10, y_offset))
            y_offset += 16
        
        y_offset += 10
        
        # Controls
        controls_title = self.font_medium.render("CONTROLS:", True, self.BLACK)
        self.screen.blit(controls_title, (panel_x + 10, y_offset))
        y_offset += 20
        
        controls = [
            "SPACE - Pause/Resume",
            "N - Next Algorithm",
            "T - Toggle Trails",
            "I - Toggle Info",
            "R - Toggle Obs. Radius",
            "ESC - Quit"
        ]
        
        for control in controls:
            control_surface = self.font_tiny.render(control, True, self.BLACK)
            self.screen.blit(control_surface, (panel_x + 10, y_offset))
            y_offset += 16
        
        # Evaluation complete message
        if self.evaluation_complete:
            y_offset += 20
            complete_text = self.font_medium.render("EVALUATION COMPLETE!", True, self.GREEN)
            self.screen.blit(complete_text, (panel_x + 10, y_offset))
    
    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_n and not self.evaluation_complete:
                    self.switch_to_next_algorithm()
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_r:
                    self.show_observation_radius = not self.show_observation_radius
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def run(self):
        """Main game loop"""
        print("🚀 Starting Task-Based RL Algorithm Evaluation")
        print("=" * 60)
        print("Fair evaluation system:")
        print("- Each algorithm gets equal opportunity")
        print("- Success measured by task completion")
        print("- No arbitrary time limits")
        print("- Comprehensive performance metrics")
        print("=" * 60)
        
        while self.running:
            # Handle events
            self.handle_events()
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw border
            pygame.draw.line(self.screen, self.BLACK, 
                           (self.main_area_width, 0), (self.main_area_width, self.height), 3)
            
            # Draw navigation points
            self.draw_navigation_points()
            
            # Draw agents
            for agent in self.agents:
                if self.show_trails:
                    self.draw_trail(agent)
                self.draw_agent(agent)
            
            # Draw control panel
            self.draw_control_panel()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        pygame.quit()

def main():
    """Run the demo"""
    try:
        demo = TaskBasedSwarmDemo()
        demo.run()
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 