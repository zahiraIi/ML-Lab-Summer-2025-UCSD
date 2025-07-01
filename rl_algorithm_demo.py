#!/usr/bin/env python3
"""
RL Algorithm Comparison Demo - Decentralized Swarm Navigation
Shows DDPG, SAC, TD3, and PPO algorithms competing in real-time
Each bot has limited local knowledge (decentralized)
"""

import pygame
import numpy as np
import torch
import torch.nn as nn
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import json
from collections import deque
import matplotlib.pyplot as plt

# Import RL algorithms
from stable_baselines3 import DDPG, SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

# Import our implementations
from enhanced_multibot_env import EnhancedMultiBotEnv
from step3_decentralized_swarm import DecentralizedSwarmLearner, DecentralizedObservationSpace

@dataclass
class AlgorithmConfig:
    """Configuration for each RL algorithm"""
    name: str
    color: Tuple[int, int, int]
    model_class: Any
    hyperparams: Dict
    is_trained: bool = False
    performance_score: float = 0.0
    success_rate: float = 0.0

class DecentralizedMultiAlgorithmDemo:
    """Demo showing multiple RL algorithms competing in decentralized swarm navigation"""
    
    def __init__(self, width: int = 1400, height: int = 900):
        pygame.init()
        
        # Display setup
        self.width = width
        self.height = height
        self.control_panel_width = 350
        self.main_area_width = width - self.control_panel_width
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Decentralized Multi-Algorithm Swarm Demo")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.PURPLE = (128, 0, 128)
        self.ORANGE = (255, 165, 0)
        self.CYAN = (0, 255, 255)
        self.GRAY = (128, 128, 128)
        
        # Fonts
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Algorithm configurations
        self.algorithms = {
            'DDPG': AlgorithmConfig(
                name='DDPG',
                color=self.BLUE,
                model_class=DDPG,
                hyperparams={'learning_rate': 1e-3, 'buffer_size': 100000}
            ),
            'SAC': AlgorithmConfig(
                name='SAC',
                color=self.RED,
                model_class=SAC,
                hyperparams={'learning_rate': 3e-4, 'buffer_size': 100000}
            ),
            'TD3': AlgorithmConfig(
                name='TD3',
                color=self.GREEN,
                model_class=TD3,
                hyperparams={'learning_rate': 1e-3, 'buffer_size': 100000}
            ),
            'PPO': AlgorithmConfig(
                name='PPO',
                color=self.PURPLE,
                model_class=PPO,
                hyperparams={'learning_rate': 3e-4, 'n_steps': 2048}
            )
        }
        
        # Current algorithm being tested
        self.current_algorithm = 'DDPG'
        self.algorithm_cycle_time = 10.0  # Seconds to test each algorithm
        self.last_algorithm_switch = time.time()
        
        # Navigation setup
        self.start_point = np.array([150.0, self.height/2])
        self.target_point = np.array([self.main_area_width - 150, self.height/2])
        
        # Swarm setup
        self.num_agents = 5
        self.observation_radius = 100.0  # Visual radius for decentralized observation
        self.agents = []
        self.agent_trails = {}
        self.max_trail_length = 50
        
        # Physics and environment
        self.env = None
        self.decentralized_learner = None
        self.setup_environment()
        
        # Control flags
        self.running = True
        self.paused = False
        self.show_trails = True
        self.show_info = True
        self.show_metrics = True
        self.show_observation_radius = True
        self.auto_cycle = True
        
        # Timing
        self.clock = pygame.time.Clock()
        self.dt = 0.1
        
        # Performance tracking
        self.performance_history = {algo: deque(maxlen=100) for algo in self.algorithms.keys()}
        self.current_episode_data = {
            'start_time': time.time(),
            'distances': [],
            'formation_errors': [],
            'actions': []
        }
        
        # Training status
        self.training_status = "Initializing..."
        self.episodes_completed = 0
        self.training_thread = None
        self.start_training()
    
    def setup_environment(self):
        """Set up the decentralized environment"""
        # Create environment for current algorithm test
        self.env = EnhancedMultiBotEnv(
            num_bots=self.num_agents,
            task="navigation",
            reward_type="dense",
            dt=self.dt
        )
        
        # Create decentralized learner
        self.decentralized_learner = DecentralizedSwarmLearner(
            env=self.env,
            observation_radius=2.5,  # Physics radius
            learning_rate=3e-4
        )
        
        # Initialize agents with random positions near start
        self.agents = []
        for i in range(self.num_agents):
            angle = (i / self.num_agents) * 2 * np.pi
            offset = np.array([np.cos(angle), np.sin(angle)]) * 30
            position = self.start_point + offset + np.random.normal(0, 10, 2)
            
            agent = {
                'id': i,
                'position': position,
                'velocity': np.array([0.0, 0.0]),
                'spinning_speed': 0.0,
                'color': self.algorithms[self.current_algorithm].color,
                'trail': deque(maxlen=self.max_trail_length),
                'neighbors': [],
                'local_target': self.target_point.copy()
            }
            agent['trail'].append(position.copy())
            self.agents.append(agent)
        
        # Reset environment
        obs, _ = self.env.reset()
    
    def start_training(self):
        """Start background training thread"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
            self.training_thread.start()
    
    def training_loop(self):
        """Background training loop for current algorithm"""
        while self.running:
            try:
                if not self.paused:
                    self.run_training_episode()
                time.sleep(0.01)  # Small delay to prevent overwhelming
            except Exception as e:
                print(f"Training error: {e}")
                time.sleep(1.0)
    
    def run_training_episode(self):
        """Run one training episode for current algorithm"""
        # Get current state and convert to local observations
        current_state = np.array([agent['position'] for agent in self.agents]).flatten()
        self.decentralized_learner.update_velocity_tracking(current_state)
        
        # Get local observations (decentralized - each agent only sees nearby neighbors)
        local_observations = self.get_decentralized_observations()
        
        # Select actions using current algorithm policy
        actions = self.decentralized_learner.select_actions(
            local_observations, 
            exploration_noise=0.1
        )
        
        # Apply physics-based movement
        self.update_agent_physics(actions)
        
        # Calculate decentralized rewards
        rewards = self.calculate_decentralized_rewards(actions)
        
        # Store performance data
        self.update_performance_tracking(actions, rewards)
        
        # Update policy (simplified for demo)
        if len(self.decentralized_learner.replay_buffer) > 100:
            self.decentralized_learner.update_policy()
        
        self.episodes_completed += 1
    
    def get_decentralized_observations(self) -> List[torch.Tensor]:
        """Get decentralized observations - each agent only sees nearby neighbors"""
        observations = []
        
        for i, agent in enumerate(self.agents):
            obs = np.zeros(20)  # Local observation vector
            
            # Own position relative to target (limited knowledge)
            target_dir = self.target_point - agent['position']
            target_dist = np.linalg.norm(target_dir)
            obs[0:2] = target_dir / (target_dist + 1e-6) if target_dist > 0 else [0, 0]
            obs[2] = min(target_dist / 500.0, 1.0)  # Normalized distance
            
            # Own velocity
            obs[3:5] = agent['velocity']
            
            # Find neighbors within observation radius
            neighbors = []
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    dist = np.linalg.norm(other_agent['position'] - agent['position'])
                    if dist < self.observation_radius:
                        neighbors.append({
                            'id': j,
                            'distance': dist,
                            'relative_pos': other_agent['position'] - agent['position'],
                            'velocity': other_agent['velocity']
                        })
            
            # Sort by distance and take closest 3 neighbors
            neighbors.sort(key=lambda x: x['distance'])
            agent['neighbors'] = neighbors[:3]
            
            # Add neighbor information to observation
            obs_idx = 5
            for neighbor in agent['neighbors']:
                if obs_idx + 4 < len(obs):
                    obs[obs_idx:obs_idx+2] = neighbor['relative_pos'] / self.observation_radius
                    obs[obs_idx+2:obs_idx+4] = neighbor['velocity'] * 0.1
                    obs_idx += 4
            
            observations.append(torch.FloatTensor(obs))
        
        return observations
    
    def update_agent_physics(self, actions: np.ndarray):
        """Update agent positions based on actions (simplified physics)"""
        for i, agent in enumerate(self.agents):
            # Convert spinning speed to movement force
            action = actions[i]
            
            # Direction towards target (with some noise for realism)
            target_dir = self.target_point - agent['position']
            target_dist = np.linalg.norm(target_dir)
            
            if target_dist > 10.0:
                direction = target_dir / target_dist
                
                # Add formation keeping force based on neighbors
                formation_force = np.array([0.0, 0.0])
                if agent['neighbors']:
                    for neighbor in agent['neighbors']:
                        # Maintain spacing
                        if neighbor['distance'] < 40:
                            repulsion = -neighbor['relative_pos'] / (neighbor['distance'] + 1e-6)
                            formation_force += repulsion * 0.1
                
                # Combine forces
                total_force = direction * abs(action) * 20.0 + formation_force
                
                # Update velocity and position
                agent['velocity'] = agent['velocity'] * 0.9 + total_force * self.dt
                agent['position'] += agent['velocity'] * self.dt
                
                # Keep in bounds
                agent['position'][0] = np.clip(agent['position'][0], 20, self.main_area_width - 20)
                agent['position'][1] = np.clip(agent['position'][1], 20, self.height - 20)
            
            agent['spinning_speed'] = action
            
            # Update trail
            agent['trail'].append(agent['position'].copy())
    
    def calculate_decentralized_rewards(self, actions: np.ndarray) -> List[float]:
        """Calculate rewards for each agent based on local information"""
        rewards = []
        
        for i, agent in enumerate(self.agents):
            reward = 0.0
            
            # Distance to target reward
            target_dist = np.linalg.norm(agent['position'] - self.target_point)
            reward -= target_dist * 0.01
            
            # Neighbor coordination reward
            if agent['neighbors']:
                avg_neighbor_dist = np.mean([n['distance'] for n in agent['neighbors']])
                reward += 0.1 if 30 < avg_neighbor_dist < 80 else -0.05
            
            # Action penalty (energy efficiency)
            reward -= abs(actions[i]) * 0.1
            
            # Success bonus
            if target_dist < 50:
                reward += 10.0
            
            rewards.append(reward)
        
        return rewards
    
    def update_performance_tracking(self, actions: np.ndarray, rewards: List[float]):
        """Update performance metrics for current algorithm"""
        # Calculate swarm center
        swarm_center = np.mean([agent['position'] for agent in self.agents], axis=0)
        distance_to_target = np.linalg.norm(swarm_center - self.target_point)
        
        # Formation error
        positions = np.array([agent['position'] for agent in self.agents])
        formation_error = np.var(positions[:, 1])  # Y-axis variance
        
        # Store data
        self.current_episode_data['distances'].append(distance_to_target)
        self.current_episode_data['formation_errors'].append(formation_error)
        self.current_episode_data['actions'].append(np.mean(np.abs(actions)))
        
        # Update algorithm performance
        avg_reward = np.mean(rewards)
        self.performance_history[self.current_algorithm].append(avg_reward)
        
        # Calculate success rate
        success = distance_to_target < 50
        current_successes = sum(1 for d in list(self.current_episode_data['distances'])[-20:] if d < 50)
        self.algorithms[self.current_algorithm].success_rate = current_successes / min(20, len(self.current_episode_data['distances']))
        
        # Update overall performance score
        self.algorithms[self.current_algorithm].performance_score = np.mean(list(self.performance_history[self.current_algorithm])[-50:]) if self.performance_history[self.current_algorithm] else 0
    
    def cycle_algorithm(self):
        """Switch to next algorithm"""
        algo_names = list(self.algorithms.keys())
        current_idx = algo_names.index(self.current_algorithm)
        next_idx = (current_idx + 1) % len(algo_names)
        
        self.current_algorithm = algo_names[next_idx]
        self.last_algorithm_switch = time.time()
        
        # Update agent colors
        for agent in self.agents:
            agent['color'] = self.algorithms[self.current_algorithm].color
        
        # Reset episode data
        self.current_episode_data = {
            'start_time': time.time(),
            'distances': [],
            'formation_errors': [],
            'actions': []
        }
        
        # Reset positions
        self.setup_environment()
        
        self.training_status = f"Testing {self.current_algorithm}"
    
    def draw_agent(self, agent: Dict):
        """Draw individual agent with decentralized information"""
        pos = tuple(map(int, agent['position']))
        
        # Draw observation radius if enabled
        if self.show_observation_radius:
            pygame.draw.circle(self.screen, (*agent['color'], 30), pos, int(self.observation_radius), 1)
        
        # Draw agent body
        pygame.draw.circle(self.screen, agent['color'], pos, 12)
        pygame.draw.circle(self.screen, self.BLACK, pos, 12, 2)
        
        # Draw direction indicator
        if np.linalg.norm(agent['velocity']) > 0.1:
            direction = agent['velocity'] / (np.linalg.norm(agent['velocity']) + 1e-6)
            end_pos = pos + direction * 20
            pygame.draw.line(self.screen, self.BLACK, pos, tuple(map(int, end_pos)), 2)
        
        # Draw spinning speed indicator
        spin_radius = abs(agent['spinning_speed']) * 3 + 3
        spin_color = self.RED if agent['spinning_speed'] > 0 else self.BLUE
        pygame.draw.circle(self.screen, spin_color, pos, int(spin_radius), 1)
        
        # Draw connections to neighbors
        for neighbor in agent['neighbors']:
            neighbor_agent = self.agents[neighbor['id']]
            neighbor_pos = tuple(map(int, neighbor_agent['position']))
            pygame.draw.line(self.screen, self.GRAY, pos, neighbor_pos, 1)
        
        # Draw agent info
        if self.show_info:
            info_text = f"{agent['id']}: ω={agent['spinning_speed']:.1f}"
            text_surface = self.font_small.render(info_text, True, self.BLACK)
            text_pos = (pos[0] - 25, pos[1] - 30)
            self.screen.blit(text_surface, text_pos)
    
    def draw_trail(self, agent: Dict):
        """Draw agent trail"""
        if not self.show_trails or len(agent['trail']) < 2:
            return
        
        trail_points = list(agent['trail'])
        for i in range(1, len(trail_points)):
            alpha = i / len(trail_points)
            color = tuple(int(c * alpha) for c in agent['color'])
            
            start_pos = tuple(map(int, trail_points[i-1]))
            end_pos = tuple(map(int, trail_points[i]))
            
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
    
    def draw_navigation_points(self):
        """Draw start and target points"""
        # Start point (A)
        start_pos = tuple(map(int, self.start_point))
        pygame.draw.circle(self.screen, self.GREEN, start_pos, 25)
        pygame.draw.circle(self.screen, self.BLACK, start_pos, 25, 3)
        start_text = self.font_large.render("A", True, self.BLACK)
        text_rect = start_text.get_rect(center=start_pos)
        self.screen.blit(start_text, text_rect)
        
        # Target point (B)
        target_pos = tuple(map(int, self.target_point))
        pygame.draw.circle(self.screen, self.RED, target_pos, 25)
        pygame.draw.circle(self.screen, self.BLACK, target_pos, 25, 3)
        target_text = self.font_large.render("B", True, self.WHITE)
        text_rect = target_text.get_rect(center=target_pos)
        self.screen.blit(target_text, text_rect)
        
        # Draw direct path
        pygame.draw.line(self.screen, self.GRAY, start_pos, target_pos, 2)
    
    def draw_control_panel(self):
        """Draw control panel with algorithm comparison"""
        panel_x = self.main_area_width
        panel_rect = pygame.Rect(panel_x, 0, self.control_panel_width, self.height)
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)
        pygame.draw.line(self.screen, self.BLACK, (panel_x, 0), (panel_x, self.height), 2)
        
        y_offset = 10
        line_height = 22
        
        # Title
        title = self.font_large.render("RL ALGORITHM DEMO", True, self.BLACK)
        self.screen.blit(title, (panel_x + 5, y_offset))
        y_offset += 40
        
        # Current algorithm
        current_text = self.font_medium.render(f"Current: {self.current_algorithm}", True, self.algorithms[self.current_algorithm].color)
        self.screen.blit(current_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Time remaining
        if self.auto_cycle:
            time_remaining = self.algorithm_cycle_time - (time.time() - self.last_algorithm_switch)
            time_text = self.font_small.render(f"Next switch: {time_remaining:.1f}s", True, self.BLACK)
            self.screen.blit(time_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Algorithm comparison
        comparison_title = self.font_medium.render("ALGORITHM PERFORMANCE:", True, self.BLACK)
        self.screen.blit(comparison_title, (panel_x + 10, y_offset))
        y_offset += 25
        
        for algo_name, config in self.algorithms.items():
            # Algorithm name and color indicator
            color_rect = pygame.Rect(panel_x + 10, y_offset, 15, 15)
            pygame.draw.rect(self.screen, config.color, color_rect)
            pygame.draw.rect(self.screen, self.BLACK, color_rect, 1)
            
            # Performance text
            perf_text = f"{algo_name}: {config.performance_score:.2f}"
            success_text = f"Success: {config.success_rate:.1%}"
            
            text_surface = self.font_small.render(perf_text, True, self.BLACK)
            self.screen.blit(text_surface, (panel_x + 30, y_offset))
            
            success_surface = self.font_small.render(success_text, True, self.BLACK)
            self.screen.blit(success_surface, (panel_x + 30, y_offset + 12))
            
            y_offset += 35
        
        y_offset += 15
        
        # Current metrics
        if self.show_metrics:
            swarm_center = np.mean([agent['position'] for agent in self.agents], axis=0)
            distance_to_target = np.linalg.norm(swarm_center - self.target_point)
            
            metrics_text = [
                "CURRENT METRICS:",
                f"Distance to Target: {distance_to_target:.1f}",
                f"Episodes: {self.episodes_completed}",
                f"Training Status: {self.training_status}",
                f"Observation Radius: {self.observation_radius:.0f}px",
                "",
                "DECENTRALIZED FEATURES:",
                "✓ Local observations only",
                "✓ Shared policy network", 
                "✓ Neighbor communication",
                "✓ No global knowledge",
                "",
                "CONTROLS:",
                "SPACE - Pause/Resume",
                "N - Next Algorithm",
                "A - Toggle Auto-cycle",
                "T - Toggle Trails",
                "I - Toggle Info",
                "R - Toggle Observation Radius",
                "ESC - Quit"
            ]
            
            for text in metrics_text:
                if text.startswith("CURRENT METRICS:") or text.startswith("DECENTRALIZED FEATURES:") or text.startswith("CONTROLS:"):
                    surface = self.font_medium.render(text, True, self.BLACK)
                else:
                    surface = self.font_small.render(text, True, self.BLACK)
                self.screen.blit(surface, (panel_x + 10, y_offset))
                y_offset += line_height
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_n:
                    self.cycle_algorithm()
                elif event.key == pygame.K_a:
                    self.auto_cycle = not self.auto_cycle
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_r:
                    self.show_observation_radius = not self.show_observation_radius
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    if mouse_pos[0] < self.main_area_width:  # Only in main area
                        self.target_point = mouse_pos
    
    def run(self):
        """Main game loop"""
        while self.running:
            # Handle events
            self.handle_events()
            
            # Auto-cycle algorithms
            if self.auto_cycle and not self.paused:
                if time.time() - self.last_algorithm_switch > self.algorithm_cycle_time:
                    self.cycle_algorithm()
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw main area border
            pygame.draw.line(self.screen, self.BLACK, (self.main_area_width, 0), (self.main_area_width, self.height), 2)
            
            # Draw navigation points
            self.draw_navigation_points()
            
            # Draw agents and trails
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
    """Run the RL algorithm comparison demo"""
    print("🚀 Starting RL Algorithm Comparison Demo")
    print("=" * 50)
    print("This demo shows different RL algorithms competing")
    print("in a decentralized swarm navigation task.")
    print("Each robot has limited local knowledge!")
    print("=" * 50)
    
    demo = DecentralizedMultiAlgorithmDemo()
    demo.run()

if __name__ == "__main__":
    main() 