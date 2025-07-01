#!/usr/bin/env python3
"""
Optimized swarm navigation system with pygame visualization
Based on https://github.com/zahiraIi/swarmproject.git structure
"""

import pygame
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

# Use existing physics from notebooks
from multibot_cluster_env import MultiBotClusterEnv

@dataclass
class Agent:
    """Individual agent with physics-based movement"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    spinning_speed: float
    color: Tuple[int, int, int]
    trail: List[np.ndarray]
    
    def __post_init__(self):
        if not self.trail:
            self.trail = []

class SwarmVisualization:
    """
    Pygame-based swarm visualization similar to https://github.com/zahiraIi/swarmproject.git
    """
    
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        
        # Display settings
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Robotic Swarm Navigation - Point A to Point B")
        
        # Colors (similar to original repo style)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 100, 100)
        self.GREEN = (100, 255, 100)
        self.BLUE = (100, 100, 255)
        self.YELLOW = (255, 255, 100)
        self.CYAN = (100, 255, 255)
        self.PURPLE = (255, 100, 255)
        self.GRAY = (128, 128, 128)
        self.ORANGE = (255, 165, 0)
        
        # Agent colors
        self.agent_colors = [
            self.RED, self.BLUE, self.GREEN, self.YELLOW, 
            self.CYAN, self.PURPLE, self.ORANGE
        ]
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Simulation state
        self.running = True
        self.paused = False
        self.show_trails = True
        self.show_forces = False
        self.show_info = True
        self.show_metrics = True
        
        # Navigation points
        self.start_point = np.array([100, 400])
        self.target_point = np.array([1000, 300])
        
        # Agents
        self.agents: List[Agent] = []
        self.max_trail_length = 50
        
        # Metrics tracking
        self.metrics = {
            'time': 0.0,
            'distance_to_target': 0.0,
            'avg_spinning_speed': 0.0,
            'formation_error': 0.0,
            'energy_consumption': 0.0,
            'path_efficiency': 0.0
        }
        
        # Physics environment (using notebook physics)
        self.physics_env = None
        self.dt = 0.05
        
        # Control panel
        self.control_panel_width = 300
        self.main_area_width = width - self.control_panel_width
        
    def initialize_agents(self, num_agents: int = 5):
        """Initialize agents near start point"""
        self.agents = []
        
        # Create agents in line formation near start point
        spacing = 40
        start_x = self.start_point[0]
        start_y = self.start_point[1]
        
        for i in range(num_agents):
            y_offset = (i - num_agents//2) * spacing
            position = np.array([start_x, start_y + y_offset], dtype=float)
            velocity = np.array([0.0, 0.0])
            
            agent = Agent(
                id=i,
                position=position,
                velocity=velocity,
                spinning_speed=0.0,
                color=self.agent_colors[i % len(self.agent_colors)],
                trail=[]
            )
            
            self.agents.append(agent)
        
        # Initialize physics environment
        self.setup_physics_environment()
    
    def setup_physics_environment(self):
        """Set up the physics environment for navigation task"""
        num_bots = len(self.agents)
        
        # Initialize physics positions
        initial_positions = []
        physics_target = self.screen_to_physics(self.target_point)
        
        for agent in self.agents:
            physics_pos = self.screen_to_physics(agent.position)
            initial_positions.extend([physics_pos[0], physics_pos[1]])
        
        self.physics_env.X0 = np.array(initial_positions)
        
        # Enhanced navigation reward function with proper target tracking
        def navigation_reward(X):
            P = X.reshape(num_bots, 2)
            com = P.mean(axis=0)
            
            # Distance to target (main objective)
            target_dist = np.linalg.norm(com - physics_target)
            distance_reward = -5.0 * target_dist  # Strong penalty for being far from target
            
            # Formation bonus (maintain line formation)
            formation_error = np.var(P[:, 1])  # Variance in y-direction
            formation_reward = -0.5 * formation_error
            
            # Progress bonus (reward for moving towards target)
            # This encourages continuous movement toward target
            previous_com = getattr(self, '_prev_com', com)
            if hasattr(self, '_prev_com'):
                prev_dist = np.linalg.norm(previous_com - physics_target)
                curr_dist = target_dist
                progress = prev_dist - curr_dist  # Positive if moving closer
                progress_reward = 10.0 * progress
            else:
                progress_reward = 0.0
            self._prev_com = com.copy()
            
            # Success bonus
            success_bonus = 100.0 if target_dist < 0.3 else 0
            
            # Penalty for being too far apart (cohesion)
            max_spread = np.max([np.linalg.norm(P[i] - com) for i in range(num_bots)])
            cohesion_penalty = -1.0 * max_spread if max_spread > 2.0 else 0
            
            total_reward = distance_reward + formation_reward + progress_reward + success_bonus + cohesion_penalty
            
            return total_reward
        
        self.physics_env._reward = navigation_reward
        
        # Reset environment with new target
        obs, _ = self.physics_env.reset()
    
    def screen_to_physics(self, screen_pos: np.ndarray) -> np.ndarray:
        """Convert screen coordinates to physics coordinates"""
        # Scale screen coordinates to physics range (roughly -5 to 5)
        physics_x = (screen_pos[0] - self.main_area_width/2) / (self.main_area_width/10)
        physics_y = (screen_pos[1] - self.height/2) / (self.height/10)
        return np.array([physics_x, physics_y])
    
    def physics_to_screen(self, physics_pos: np.ndarray) -> np.ndarray:
        """Convert physics coordinates to screen coordinates"""
        screen_x = physics_pos[0] * (self.main_area_width/10) + self.main_area_width/2
        screen_y = physics_pos[1] * (self.height/10) + self.height/2
        return np.array([screen_x, screen_y])
    
    def update_physics(self, actions: Optional[np.ndarray] = None):
        """Update physics simulation"""
        if self.physics_env is None:
            return
        
        if actions is None:
            # Enhanced navigation policy: proper force-based movement towards target
            current_state = self.physics_env.state.reshape(len(self.agents), 2)
            target_physics = self.screen_to_physics(self.target_point)
            
            actions = np.zeros(len(self.agents))
            
            # Calculate center of mass for formation control
            com = np.mean(current_state, axis=0)
            target_direction = target_physics - com
            target_distance = np.linalg.norm(target_direction)
            
            # Normalize direction
            if target_distance > 0.1:
                target_direction = target_direction / target_distance
            else:
                target_direction = np.array([0.0, 0.0])
            
            for i, agent_pos in enumerate(current_state):
                # Individual agent navigation force
                individual_target_dir = target_physics - agent_pos
                individual_distance = np.linalg.norm(individual_target_dir)
                
                # Formation keeping (maintain line formation)
                formation_offset = np.array([0.0, (i - len(self.agents)/2) * 0.5])
                desired_pos = com + target_direction * 0.5 + formation_offset
                formation_dir = desired_pos - agent_pos
                
                # Combine navigation and formation forces
                nav_force = 2.0 * individual_target_dir if individual_distance > 0.1 else np.array([0.0, 0.0])
                formation_force = 1.0 * formation_dir
                
                # Total force
                total_force = nav_force + formation_force
                
                # Convert force to spinning speed (simplified model)
                # In real robotics, this would be more complex force-to-torque mapping
                force_magnitude = np.linalg.norm(total_force)
                actions[i] = np.clip(force_magnitude * 0.3, 0.1, 2.0)
                
                # Add some variation to avoid getting stuck
                if target_distance < 0.2:  # Near target
                    actions[i] *= 0.1  # Slow down
                else:
                    actions[i] += 0.1 * np.sin(self.metrics['time'] * 1.5 + i * 2)
        
        # Step physics
        obs, reward, terminated, truncated, info = self.physics_env.step(actions)
        
        # Update agent positions from physics
        physics_positions = self.physics_env.state.reshape(len(self.agents), 2)
        
        for i, agent in enumerate(self.agents):
            old_pos = agent.position.copy()
            agent.position = self.physics_to_screen(physics_positions[i])
            agent.velocity = (agent.position - old_pos) / self.dt if self.dt > 0 else np.array([0.0, 0.0])
            agent.spinning_speed = actions[i]
            
            # Update trail
            agent.trail.append(agent.position.copy())
            if len(agent.trail) > self.max_trail_length:
                agent.trail.pop(0)
        
        # Update metrics
        self.update_metrics(actions)
    
    def update_metrics(self, actions: np.ndarray):
        """Update performance metrics"""
        if not self.agents:
            return
        
        # Center of mass
        com = np.mean([agent.position for agent in self.agents], axis=0)
        
        # Distance to target
        self.metrics['distance_to_target'] = np.linalg.norm(com - self.target_point)
        
        # Average spinning speed
        self.metrics['avg_spinning_speed'] = np.mean(np.abs(actions))
        
        # Formation error (variance in y-direction for line formation)
        y_positions = [agent.position[1] for agent in self.agents]
        self.metrics['formation_error'] = np.var(y_positions)
        
        # Energy consumption
        self.metrics['energy_consumption'] += np.sum(actions**2) * self.dt
        
        # Path efficiency (if we have trail data)
        if len(self.agents[0].trail) > 1:
            actual_path = sum([
                np.linalg.norm(self.agents[0].trail[i] - self.agents[0].trail[i-1])
                for i in range(1, len(self.agents[0].trail))
            ])
            direct_distance = np.linalg.norm(self.target_point - self.start_point)
            self.metrics['path_efficiency'] = direct_distance / (actual_path + 1e-6)
        
        self.metrics['time'] += self.dt
    
    def draw_agent(self, agent: Agent):
        """Draw individual agent with spinning speed"""
        pos = tuple(map(int, agent.position))
        
        # Draw agent body
        pygame.draw.circle(self.screen, agent.color, pos, 15)
        pygame.draw.circle(self.screen, self.BLACK, pos, 15, 2)
        
        # Draw direction indicator
        if np.linalg.norm(agent.velocity) > 0.1:
            direction = agent.velocity / (np.linalg.norm(agent.velocity) + 1e-6)
            end_pos = pos + direction * 20
            pygame.draw.line(self.screen, self.BLACK, pos, tuple(map(int, end_pos)), 3)
        
        # Draw spinning speed indicator
        spin_radius = abs(agent.spinning_speed) * 5 + 5
        spin_color = self.RED if agent.spinning_speed > 0 else self.BLUE
        pygame.draw.circle(self.screen, spin_color, pos, int(spin_radius), 1)
        
        # Draw agent ID and spinning speed
        if self.show_info:
            speed_text = f"{agent.id}: ω={agent.spinning_speed:.2f}"
            text_surface = self.font_small.render(speed_text, True, self.BLACK)
            text_pos = (pos[0] - 25, pos[1] - 35)
            self.screen.blit(text_surface, text_pos)
    
    def draw_trail(self, agent: Agent):
        """Draw agent trail"""
        if len(agent.trail) < 2:
            return
        
        # Draw trail with fading effect
        for i in range(1, len(agent.trail)):
            alpha = i / len(agent.trail)
            color = tuple(int(c * alpha) for c in agent.color)
            
            start_pos = tuple(map(int, agent.trail[i-1]))
            end_pos = tuple(map(int, agent.trail[i]))
            
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
    
    def draw_navigation_points(self):
        """Draw start and target points"""
        # Start point (A)
        start_pos = tuple(map(int, self.start_point))
        pygame.draw.circle(self.screen, self.GREEN, start_pos, 20)
        pygame.draw.circle(self.screen, self.BLACK, start_pos, 20, 3)
        start_text = self.font_medium.render("A", True, self.BLACK)
        text_rect = start_text.get_rect(center=start_pos)
        self.screen.blit(start_text, text_rect)
        
        # Target point (B)
        target_pos = tuple(map(int, self.target_point))
        pygame.draw.circle(self.screen, self.RED, target_pos, 20)
        pygame.draw.circle(self.screen, self.BLACK, target_pos, 20, 3)
        target_text = self.font_medium.render("B", True, self.WHITE)
        text_rect = target_text.get_rect(center=target_pos)
        self.screen.blit(target_text, text_rect)
        
        # Draw direct path
        pygame.draw.line(self.screen, self.GRAY, start_pos, target_pos, 2)
    
    def draw_control_panel(self):
        """Draw control panel (similar to original repo style)"""
        panel_x = self.main_area_width
        panel_rect = pygame.Rect(panel_x, 0, self.control_panel_width, self.height)
        pygame.draw.rect(self.screen, (240, 240, 240), panel_rect)
        pygame.draw.line(self.screen, self.BLACK, (panel_x, 0), (panel_x, self.height), 2)
        
        y_offset = 20
        line_height = 25
        
        # Title
        title = self.font_large.render("SWARM CONTROL", True, self.BLACK)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 50
        
        # Metrics
        if self.show_metrics:
            metrics_text = [
                f"Time: {self.metrics['time']:.1f}s",
                f"Distance to Target: {self.metrics['distance_to_target']:.1f}",
                f"Avg Spin Speed: {self.metrics['avg_spinning_speed']:.2f}",
                f"Formation Error: {self.metrics['formation_error']:.2f}",
                f"Energy Used: {self.metrics['energy_consumption']:.1f}",
                f"Path Efficiency: {self.metrics['path_efficiency']:.3f}",
            ]
            
            for text in metrics_text:
                surface = self.font_small.render(text, True, self.BLACK)
                self.screen.blit(surface, (panel_x + 10, y_offset))
                y_offset += line_height
        
        y_offset += 20
        
        # Controls
        controls_text = [
            "CONTROLS:",
            "SPACE - Pause/Resume",
            "T - Toggle Trails",
            "I - Toggle Info",
            "M - Toggle Metrics",
            "R - Reset Simulation",
            "Q - Quit",
            "",
            "AGENTS:",
            f"Count: {len(self.agents)}",
            f"Formation: Line",
            f"Task: Point A → B"
        ]
        
        for text in controls_text:
            if text == "CONTROLS:" or text == "AGENTS:":
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
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                elif event.key == pygame.K_m:
                    self.show_metrics = not self.show_metrics
                elif event.key == pygame.K_r:
                    self.reset_simulation()
                elif event.key == pygame.K_q:
                    self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    if mouse_pos[0] < self.main_area_width:  # Only in main area
                        self.target_point = mouse_pos
                        self.setup_physics_environment()  # Reinitialize with new target
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.metrics = {
            'time': 0.0,
            'distance_to_target': 0.0,
            'avg_spinning_speed': 0.0,
            'formation_error': 0.0,
            'energy_consumption': 0.0,
            'path_efficiency': 0.0
        }
        
        # Reset agent positions
        for agent in self.agents:
            agent.trail = []
        
        self.initialize_agents(len(self.agents))
    
    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        fps = 60
        
        print("🤖 Swarm Navigation Visualization Started")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  T - Toggle Trails")
        print("  I - Toggle Info")
        print("  M - Toggle Metrics")
        print("  R - Reset")
        print("  Q - Quit")
        print("  Left Click - Set new target")
        
        while self.running:
            dt = clock.tick(fps) / 1000.0  # Convert to seconds
            
            self.handle_events()
            
            if not self.paused:
                self.update_physics()
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw main simulation area
            main_area_rect = pygame.Rect(0, 0, self.main_area_width, self.height)
            pygame.draw.rect(self.screen, self.WHITE, main_area_rect)
            
            # Draw navigation points
            self.draw_navigation_points()
            
            # Draw agent trails
            if self.show_trails:
                for agent in self.agents:
                    self.draw_trail(agent)
            
            # Draw agents
            for agent in self.agents:
                self.draw_agent(agent)
            
            # Draw control panel
            self.draw_control_panel()
            
            # Show pause indicator
            if self.paused:
                pause_text = self.font_large.render("PAUSED", True, self.RED)
                text_rect = pause_text.get_rect(center=(self.main_area_width//2, 50))
                self.screen.blit(pause_text, text_rect)
            
            pygame.display.flip()
        
        pygame.quit()

def main():
    """Main function to run the optimized swarm navigation"""
    print("🚀 Optimized Swarm Navigation System")
    print("Based on physics from notebooks + visualization style from swarmproject repo")
    print()
    
    # Create and run visualization
    viz = SwarmVisualization(width=1200, height=800)
    viz.initialize_agents(num_agents=5)
    viz.run()

if __name__ == "__main__":
    main() 