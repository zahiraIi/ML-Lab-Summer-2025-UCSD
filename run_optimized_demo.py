#!/usr/bin/env python3
"""
Main script to run the swarm navigation demo
Shows different RL algorithms with spinning speed visualization
"""

import numpy as np
from multibot_cluster_env import MultiBotClusterEnv
from quick_swarm_demo import SwarmVisualizer
import time
from stable_baselines3 import DDPG, SAC, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise
import pygame

def create_env(num_bots=3):
    """Create environment with specified number of bots"""
    return MultiBotClusterEnv(
        num_bots=num_bots,
        task="translate",  # Point-to-point navigation
        dt=0.05
    )

def train_algorithm(algo_class, env, total_timesteps=10000):
    """Train a specific algorithm"""
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )
    
    model = algo_class(
        policy="MlpPolicy",
        env=env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log="./ddpg_tensorboard/"
    )
    
    model.learn(total_timesteps=total_timesteps)
    return model

def main():
    # Initialize environment
    env = create_env(num_bots=3)
    visualizer = SwarmVisualizer(env)
    
    # Available algorithms
    algorithms = {
        'DDPG': DDPG,
        'SAC': SAC,
        'TD3': TD3,
        'PPO': PPO
    }
    
    current_algo = None
    running = True
    paused = False
    
    # Training and visualization loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                    # Switch algorithms
                    algo_name = list(algorithms.keys())[int(event.key) - pygame.K_1]
                    print(f"Switching to {algo_name}")
                    current_algo = train_algorithm(algorithms[algo_name], env)
                    env.reset()
        
        if not paused and current_algo:
            obs = env.state
            action, _ = current_algo.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            
            # Render with spinning speeds
            visualizer.render(obs, action)
            
            if done:
                env.reset()
            
            time.sleep(0.05)  # Control visualization speed
    
    pygame.quit()

if __name__ == "__main__":
    main() 