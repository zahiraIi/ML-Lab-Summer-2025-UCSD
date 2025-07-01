#!/usr/bin/env python3
"""
Step 3: Decentralized Learning - Each bot only sees neighbors within radius
Final implementation of decentralized swarm
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from typing import List, Tuple, Dict, Optional
from collections import deque
import random

from enhanced_multibot_env import EnhancedMultiBotEnv

class DecentralizedObservationSpace:
    """Creates local observations for each agent"""
    
    def __init__(self, env: EnhancedMultiBotEnv, observation_radius: float = 2.5):
        self.env = env
        self.observation_radius = observation_radius
        self.n_agents = env.N
        
        # Local observation: [own_pos(2), own_vel(2), neighbors_info(max_neighbors * 4)]
        self.max_neighbors = min(env.N - 1, 6)  # Limit neighbors for consistency
        self.local_obs_dim = 4 + self.max_neighbors * 4  # pos + vel + neighbor info
    
    def get_local_observations(self, global_state: np.ndarray, 
                             last_actions: np.ndarray = None,
                             last_velocities: np.ndarray = None) -> List[np.ndarray]:
        """Convert global state to local observations"""
        positions = global_state.reshape(self.n_agents, 2)
        
        if last_actions is None:
            last_actions = np.zeros(self.n_agents)
        if last_velocities is None:
            last_velocities = np.zeros((self.n_agents, 2))
        
        local_observations = []
        
        for i in range(self.n_agents):
            obs = np.zeros(self.local_obs_dim)
            own_pos = positions[i]
            
            # Own position and velocity
            obs[0:2] = own_pos
            obs[2:4] = last_velocities[i]
            
            # Find neighbors within observation radius
            neighbors = []
            for j in range(self.n_agents):
                if i != j:
                    neighbor_pos = positions[j]
                    distance = np.linalg.norm(neighbor_pos - own_pos)
                    
                    if distance <= self.observation_radius:
                        relative_pos = neighbor_pos - own_pos
                        neighbors.append({
                            'id': j,
                            'distance': distance,
                            'relative_pos': relative_pos,
                            'action': last_actions[j],
                            'velocity': last_velocities[j]
                        })
            
            # Sort neighbors by distance and take closest ones
            neighbors.sort(key=lambda x: x['distance'])
            neighbors = neighbors[:self.max_neighbors]
            
            # Add neighbor information to observation
            obs_idx = 4
            for neighbor in neighbors:
                obs[obs_idx:obs_idx+2] = neighbor['relative_pos']
                obs[obs_idx+2] = neighbor['action']
                obs[obs_idx+3] = neighbor['distance'] / self.observation_radius  # Normalized
                obs_idx += 4
            
            local_observations.append(obs)
        
        return local_observations

class DecentralizedActorCritic(nn.Module):
    """Shared policy network for all agents (same architecture, same weights)"""
    
    def __init__(self, obs_dim: int, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim),
            nn.Tanh()
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def forward(self, obs):
        features = self.feature_net(obs)
        action = self.actor(features) * 2.0  # Scale to [-2, 2]
        value = self.critic(features)
        return action, value

class DecentralizedSwarmLearner:
    """Decentralized learning where all agents share the same policy"""
    
    def __init__(self, 
                 env: EnhancedMultiBotEnv,
                 observation_radius: float = 2.5,
                 learning_rate: float = 3e-4,
                 shared_policy: bool = True):
        
        self.env = env
        self.n_agents = env.N
        self.observation_radius = observation_radius
        self.shared_policy = shared_policy
        
        # Create observation space
        self.obs_space = DecentralizedObservationSpace(env, observation_radius)
        
        # Create shared policy network
        self.policy_net = DecentralizedActorCritic(self.obs_space.local_obs_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience storage
        self.replay_buffer = deque(maxlen=50000)
        
        # Training parameters
        self.gamma = 0.99
        self.batch_size = 128
        self.update_frequency = 4
        
        # State tracking
        self.last_actions = np.zeros(self.n_agents)
        self.last_velocities = np.zeros((self.n_agents, 2))
        self.last_positions = np.zeros((self.n_agents, 2))
        
        # Metrics
        self.training_metrics = {
            'episode_rewards': [],
            'formation_errors': [],
            'cooperation_scores': [],
            'policy_losses': [],
            'value_losses': []
        }
    
    def get_local_observations(self, global_state: np.ndarray) -> List[torch.Tensor]:
        """Get local observations for all agents"""
        local_obs = self.obs_space.get_local_observations(
            global_state, self.last_actions, self.last_velocities
        )
        return [torch.FloatTensor(obs) for obs in local_obs]
    
    def select_actions(self, observations: List[torch.Tensor], 
                      exploration_noise: float = 0.1) -> np.ndarray:
        """Select actions using shared policy"""
        actions = []
        
        for obs in observations:
            with torch.no_grad():
                action, _ = self.policy_net(obs.unsqueeze(0))
                action = action.squeeze(0).numpy()
                
                # Add exploration noise
                if exploration_noise > 0:
                    action += np.random.normal(0, exploration_noise, action.shape)
                
                actions.append(np.clip(action[0], -2.0, 2.0))
        
        self.last_actions = np.array(actions)
        return self.last_actions
    
    def update_velocity_tracking(self, new_positions: np.ndarray):
        """Update velocity tracking for local observations"""
        if hasattr(self, 'last_positions'):
            current_positions = new_positions.reshape(self.n_agents, 2)
            self.last_velocities = (current_positions - self.last_positions) / self.env.dt
            self.last_positions = current_positions.copy()
        else:
            self.last_positions = new_positions.reshape(self.n_agents, 2)
            self.last_velocities = np.zeros((self.n_agents, 2))
    
    def compute_local_rewards(self, global_reward: float, 
                            observations: List[torch.Tensor],
                            actions: np.ndarray) -> List[float]:
        """Compute individual rewards that encourage cooperation"""
        base_reward = global_reward / self.n_agents
        local_rewards = []
        
        for i, obs in enumerate(observations):
            reward = base_reward
            
            # Local cooperation bonus
            neighbor_count = 0
            neighbor_alignment = 0
            
            # Count neighbors and measure alignment
            obs_np = obs.numpy()
            for j in range(4, len(obs_np), 4):
                if obs_np[j] != 0 or obs_np[j+1] != 0:  # Valid neighbor
                    neighbor_count += 1
                    neighbor_distance = obs_np[j+3] * self.observation_radius
                    
                    # Reward for maintaining good spacing
                    optimal_distance = 1.5
                    spacing_reward = -abs(neighbor_distance - optimal_distance) * 0.1
                    reward += spacing_reward
            
            # Bonus for having neighbors (cooperation)
            if neighbor_count > 0:
                reward += 0.1 * neighbor_count
            
            # Penalty for isolation
            if neighbor_count == 0:
                reward -= 0.5
            
            # Action smoothness
            if len(self.last_actions) > 0:
                action_change = abs(actions[i] - self.last_actions[i])
                reward -= 0.05 * action_change
            
            local_rewards.append(reward)
        
        return local_rewards
    
    def store_experience(self, observations: List[torch.Tensor],
                        actions: np.ndarray,
                        rewards: List[float],
                        next_observations: List[torch.Tensor],
                        dones: List[bool]):
        """Store experiences from all agents"""
        for i in range(self.n_agents):
            experience = {
                'obs': observations[i].clone(),
                'action': actions[i],
                'reward': rewards[i],
                'next_obs': next_observations[i].clone(),
                'done': dones[i]
            }
            self.replay_buffer.append(experience)
    
    def update_policy(self):
        """Update shared policy using experiences from all agents"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from all agents' experiences
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        observations = torch.stack([exp['obs'] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_observations = torch.stack([exp['next_obs'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])
        
        # Current policy outputs
        current_actions, current_values = self.policy_net(observations)
        current_values = current_values.squeeze()
        
        # Target values
        with torch.no_grad():
            _, next_values = self.policy_net(next_observations)
            target_values = rewards + (self.gamma * next_values.squeeze() * ~dones)
        
        # Value loss
        value_loss = nn.MSELoss()(current_values, target_values)
        
        # Policy loss (advantage-weighted)
        advantages = (target_values - current_values).detach()
        action_log_probs = -0.5 * ((current_actions.squeeze() - actions) ** 2) / 0.01
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        # Store metrics
        self.training_metrics['policy_losses'].append(policy_loss.item())
        self.training_metrics['value_losses'].append(value_loss.item())
    
    def train(self, num_episodes: int = 500, max_steps_per_episode: int = 200):
        """Train the decentralized swarm"""
        print(f"🤝 Training decentralized swarm with {self.n_agents} agents...")
        print(f"   Observation radius: {self.observation_radius}")
        print(f"   Shared policy: {self.shared_policy}")
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            self.update_velocity_tracking(state)
            observations = self.get_local_observations(state)
            
            episode_reward = 0
            episode_formation_errors = []
            step_count = 0
            
            for step in range(max_steps_per_episode):
                # Select actions
                noise_level = max(0.05, 0.3 * (1 - episode / num_episodes))
                actions = self.select_actions(observations, noise_level)
                
                # Environment step
                next_state, global_reward, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                # Update tracking
                self.update_velocity_tracking(next_state)
                next_observations = self.get_local_observations(next_state)
                
                # Compute local rewards
                local_rewards = self.compute_local_rewards(global_reward, observations, actions)
                
                # Store experience
                dones = [done] * self.n_agents
                self.store_experience(observations, actions, local_rewards, next_observations, dones)
                
                # Update policy
                if step % self.update_frequency == 0:
                    self.update_policy()
                
                # Track metrics
                episode_reward += global_reward
                if 'current_formation_error' in info:
                    episode_formation_errors.append(info['current_formation_error'])
                
                observations = next_observations
                state = next_state
                step_count += 1
                
                if done:
                    break
            
            # Store episode metrics
            self.training_metrics['episode_rewards'].append(episode_reward)
            if episode_formation_errors:
                avg_formation_error = np.mean(episode_formation_errors)
                self.training_metrics['formation_errors'].append(avg_formation_error)
                
                # Cooperation score (inverse of formation error)
                cooperation_score = 1.0 / (1.0 + avg_formation_error)
                self.training_metrics['cooperation_scores'].append(cooperation_score)
            
            # Print progress
            if episode % 50 == 0:
                avg_reward = np.mean(self.training_metrics['episode_rewards'][-50:])
                avg_error = np.mean(self.training_metrics['formation_errors'][-50:]) if self.training_metrics['formation_errors'] else 0
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Error = {avg_error:.3f}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained policy"""
        evaluation_rewards = []
        formation_errors = []
        cooperation_scores = []
        neighbor_counts = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            self.update_velocity_tracking(state)
            observations = self.get_local_observations(state)
            
            episode_reward = 0
            episode_errors = []
            episode_neighbors = []
            
            done = False
            steps = 0
            
            while not done and steps < 200:
                # Select actions (no exploration)
                actions = self.select_actions(observations, exploration_noise=0.0)
                
                # Environment step
                next_state, reward, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                episode_reward += reward
                if 'current_formation_error' in info:
                    episode_errors.append(info['current_formation_error'])
                
                # Count neighbors for cooperation analysis
                total_neighbors = 0
                for obs in observations:
                    obs_np = obs.numpy()
                    if len(obs_np) > 4:  # Check if there are neighbors
                        for i in range(4, len(obs_np), 4):
                            if obs_np[i] != 0 or obs_np[i+1] != 0:  # Valid neighbor position
                                total_neighbors += 1
                
                episode_neighbors.append(total_neighbors)
                
                # Update for next iteration
                self.update_velocity_tracking(next_state)
                observations = self.get_local_observations(next_state)
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Store episode metrics
            evaluation_rewards.append(episode_reward)
            if episode_errors:
                avg_formation_error = np.mean(episode_errors)
                formation_errors.append(avg_formation_error)
                
                # Cooperation score (inverse of formation error)
                cooperation_score = 1.0 / (1.0 + avg_formation_error)
                cooperation_scores.append(cooperation_score)
            else:
                formation_errors.append(1.0)  # Default high error
                cooperation_scores.append(0.5)  # Default medium score
            
            # Store neighbor count
            if episode_neighbors:
                neighbor_counts.append(np.mean(episode_neighbors))
            else:
                neighbor_counts.append(0)
        
        # Calculate summary metrics
        mean_reward = np.mean(evaluation_rewards)
        mean_formation_error = np.mean(formation_errors)
        mean_cooperation_score = np.mean(cooperation_scores)
        success_rate = sum(1 for r in evaluation_rewards if r > -100) / len(evaluation_rewards)
        
        return {
            'episode_rewards': evaluation_rewards,
            'formation_errors': formation_errors,
            'cooperation_scores': cooperation_scores,
            'neighbor_counts': neighbor_counts,
            'mean_reward': mean_reward,
            'mean_formation_error': mean_formation_error,
            'mean_cooperation_score': mean_cooperation_score,
            'success_rate': success_rate
        }

def main():
    """Main function for Step 3 testing"""
    print("🚀 Starting Step 3: Decentralized Learning")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {"num_bots": 3, "obs_radius": 2.0},
        {"num_bots": 5, "obs_radius": 2.5},
        {"num_bots": 7, "obs_radius": 3.0}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🤖 Testing {config['num_bots']} bots with radius {config['obs_radius']}")
        
        try:
            # Create environment
            env = EnhancedMultiBotEnv(
                num_bots=config['num_bots'],
                task="formation",
                reward_type="dense",
                formation_type="line"
            )
            
            # Create learner
            learner = DecentralizedSwarmLearner(
                env=env,
                observation_radius=config['obs_radius']
            )
            
            # Quick training
            start_time = time.time()
            learner.train(num_episodes=10, max_steps_per_episode=50)
            training_time = time.time() - start_time
            
            # Evaluate
            eval_results = learner.evaluate(num_episodes=3)
            
            results[f"{config['num_bots']}_bots"] = {
                "final_reward": eval_results['mean_reward'],
                "success_rate": eval_results['success_rate'],
                "formation_error": eval_results['mean_formation_error'],
                "cooperation_score": eval_results['mean_cooperation_score'],
                "training_time": training_time,
                "observation_radius": config['obs_radius']
            }
            
            print(f"  ✅ {config['num_bots']} bots: Reward={eval_results['mean_reward']:.2f}, Success={eval_results['success_rate']:.2f}")
            
        except Exception as e:
            print(f"  ❌ {config['num_bots']} bots: Failed - {str(e)}")
            results[f"{config['num_bots']}_bots"] = {
                "final_reward": -1000,
                "success_rate": 0.0,
                "formation_error": 1.0,
                "training_time": 0,
                "error": str(e)
            }
    
    # Save results
    Path("step3_results").mkdir(exist_ok=True)
    with open("step3_results/decentralized_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Step 3 completed! Results saved to step3_results/")
    print(f"\n📊 STEP 3 SUMMARY:")
    print("=" * 40)
    
    for bot_config, result in results.items():
        if "error" not in result:
            print(f"🤖 {bot_config}: Reward={result['final_reward']:.2f}, Success={result['success_rate']:.1%}")
        else:
            print(f"❌ {bot_config}: Failed")
    
    return results

if __name__ == "__main__":
    main() 