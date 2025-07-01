#!/usr/bin/env python3
"""
Interactive swarm visualization with spinning speed labels
Based on original multibot_cluster_env.py physics
"""

import pygame
import numpy as np
from multibot_cluster_env import MultiBotClusterEnv
import time

class SwarmVisualizer:
    def __init__(self, env, width=800, height=600):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Swarm Navigation with Spinning Speeds")
        
        # Visualization parameters
        self.width = width
        self.height = height
        self.scale = 50  # pixels per unit
        self.bot_radius = 10
        self.font = pygame.font.Font(None, 24)
        
        # Colors
        self.BACKGROUND = (20, 20, 30)
        self.BOT_COLORS = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
        self.TEXT_COLOR = (255, 255, 255)
        
    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates"""
        return (
            int(self.width/2 + pos[0] * self.scale),
            int(self.height/2 - pos[1] * self.scale)
        )
        
    def draw_bot(self, pos, color, omega):
        """Draw a bot with its spinning speed"""
        screen_pos = self.world_to_screen(pos)
        
        # Draw bot circle
        pygame.draw.circle(self.screen, color, screen_pos, self.bot_radius)
        
        # Draw spinning speed label
        speed_text = f"ω={omega:.2f}"
        text_surface = self.font.render(speed_text, True, self.TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(screen_pos[0], screen_pos[1] - 20))
        self.screen.blit(text_surface, text_rect)
        
        # Draw spinning direction indicator
        if abs(omega) > 0.1:
            direction = 1 if omega > 0 else -1
            angle = pygame.time.get_ticks() * direction * 0.01
            end_pos = (
                screen_pos[0] + np.cos(angle) * self.bot_radius,
                screen_pos[1] + np.sin(angle) * self.bot_radius
            )
            pygame.draw.line(self.screen, color, screen_pos, end_pos, 2)
    
    def render(self, state, action):
        """Render the current state with spinning speeds"""
        self.screen.fill(self.BACKGROUND)
        
        # Draw target point
        target_pos = np.array([5.0, 5.0])  # From original env
        target_screen_pos = self.world_to_screen(target_pos)
        pygame.draw.circle(self.screen, (255, 255, 0), target_screen_pos, 8)
        
        # Draw bots with spinning speeds
        positions = state.reshape(-1, 2)
        for i, (pos, omega) in enumerate(zip(positions, action)):
            self.draw_bot(pos, self.BOT_COLORS[i % len(self.BOT_COLORS)], omega)
        
        pygame.display.flip()

def main():
    """Run the optimized demo"""
    try:
        env = MultiBotClusterEnv()
        visualizer = SwarmVisualizer(env)
        visualizer.render(env.state, env.action)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure pygame is installed: pip install pygame")

if __name__ == "__main__":
    main() 