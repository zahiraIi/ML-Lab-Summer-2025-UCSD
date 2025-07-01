#!/usr/bin/env python3
"""
Simple runnable navigation test system
Tests RL algorithms on point A to point B navigation with physics from notebooks
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time
from pathlib import Path

# Use the existing multibot environment from notebooks
from multibot_cluster_env import MultiBotClusterEnv
from stable_baselines3 import DDPG, SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

class NavigationTestSystem:
    """Simple navigation test system using existing physics"""
    
    def __init__(self):
        self.results_dir = Path("navigation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Navigation scenarios
        self.scenarios = {
            'short': {'start': (-2, -1), 'target': (3, 2), 'time': 15.0},
            'medium': {'start': (-3, -2), 'target': (4, 3), 'time': 20.0},
            'long': {'start': (-4, -3), 'target': (5, 4), 'time': 25.0}
        }
        
        # Test algorithms
        self.algorithms = ['DDPG', 'SAC', 'TD3', 'PPO']
    
    def create_navigation_env(self, scenario='medium', num_bots=5):
        """Create navigation environment using existing physics"""
        scenario_config = self.scenarios[scenario]
        
        # Use the existing MultiBotClusterEnv with translation task
        env = MultiBotClusterEnv(
            num_bots=num_bots,
            dt=0.05,
            T=scenario_config['time'],
            task="translate"  # Use translate task for navigation
        )
        
        # Override target and initial positions
        env.target_point = np.array(scenario_config['target'])
        
        # Set initial positions near start point
        start = np.array(scenario_config['start'])
        spacing = 2.0 / max(1, num_bots - 1) if num_bots > 1 else 0
        xs = np.linspace(start[0] - spacing/2, start[0] + spacing/2, num_bots)
        ys = np.full(num_bots, start[1])
        env.X0 = np.vstack([xs, ys]).T.reshape(-1)
        
        # Update reward function for navigation
        def navigation_reward(X):
            P = X.reshape(num_bots, 2)
            com = P.mean(axis=0)
            
            # Distance to target (primary objective)
            target_dist = np.linalg.norm(com - env.target_point)
            distance_reward = -target_dist
            
            # Formation maintenance bonus
            formation_bonus = -0.2 * np.var(P[:, 1])  # Keep line formation
            
            # Success bonus
            success_bonus = 50.0 if target_dist < 0.5 else 0
            
            return distance_reward + formation_bonus + success_bonus
        
        env._reward = navigation_reward
        return env
    
    def train_algorithm(self, algorithm, env, timesteps=20000):
        """Train an algorithm on navigation task"""
        print(f"🤖 Training {algorithm}...")
        
        env_vec = DummyVecEnv([lambda: env])
        
        # Create algorithm
        if algorithm == 'DDPG':
            action_noise = NormalActionNoise(
                mean=np.zeros(env.N), sigma=0.1 * np.ones(env.N))
            model = DDPG('MlpPolicy', env_vec, action_noise=action_noise, 
                        learning_rate=1e-3, verbose=0)
        elif algorithm == 'SAC':
            model = SAC('MlpPolicy', env_vec, learning_rate=3e-4, verbose=0)
        elif algorithm == 'TD3':
            action_noise = NormalActionNoise(
                mean=np.zeros(env.N), sigma=0.1 * np.ones(env.N))
            model = TD3('MlpPolicy', env_vec, action_noise=action_noise, 
                       learning_rate=1e-3, verbose=0)
        elif algorithm == 'PPO':
            model = PPO('MlpPolicy', env_vec, learning_rate=3e-4, verbose=0)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Train
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        training_time = time.time() - start_time
        
        # Save model
        model_path = self.results_dir / f"{algorithm}_navigation.zip"
        model.save(str(model_path))
        
        env_vec.close()
        print(f"✅ {algorithm} trained in {training_time:.1f}s")
        
        return model, training_time
    
    def test_algorithm(self, model, algorithm, scenario='medium', num_bots=5, episodes=5):
        """Test an algorithm on navigation"""
        print(f"📊 Testing {algorithm} on {scenario} scenario...")
        
        results = []
        
        for episode in range(episodes):
            env = self.create_navigation_env(scenario, num_bots)
            obs, _ = env.reset()
            
            trajectory = []
            actions_history = []
            total_reward = 0
            
            for step in range(env.H):
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                
                # Environment step
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Store data
                positions = env.state.reshape(num_bots, 2)
                trajectory.append(positions.copy())
                actions_history.append(action.copy())
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            # Analyze episode
            trajectory = np.array(trajectory)
            com_trajectory = trajectory.mean(axis=1)
            
            start_point = np.array(self.scenarios[scenario]['start'])
            target_point = np.array(self.scenarios[scenario]['target'])
            
            final_distance = np.linalg.norm(com_trajectory[-1] - target_point)
            direct_distance = np.linalg.norm(target_point - start_point)
            
            path_length = sum([
                np.linalg.norm(com_trajectory[i+1] - com_trajectory[i])
                for i in range(len(com_trajectory)-1)
            ])
            
            episode_result = {
                'final_distance': final_distance,
                'success': final_distance < 0.5,
                'path_efficiency': direct_distance / (path_length + 1e-6),
                'total_energy': sum([np.sum(a**2) for a in actions_history]),
                'navigation_time': len(trajectory) * env.dt,
                'total_reward': total_reward,
                'trajectory': trajectory,
                'actions': np.array(actions_history)
            }
            
            results.append(episode_result)
            env.close()
        
        # Aggregate results
        aggregated = {
            'algorithm': algorithm,
            'scenario': scenario,
            'success_rate': np.mean([r['success'] for r in results]),
            'avg_final_distance': np.mean([r['final_distance'] for r in results]),
            'avg_path_efficiency': np.mean([r['path_efficiency'] for r in results]),
            'avg_energy': np.mean([r['total_energy'] for r in results]),
            'avg_time': np.mean([r['navigation_time'] for r in results]),
            'avg_reward': np.mean([r['total_reward'] for r in results]),
            'episodes': results
        }
        
        print(f"   Success Rate: {aggregated['success_rate']:.2%}")
        print(f"   Avg Final Distance: {aggregated['avg_final_distance']:.2f}")
        print(f"   Path Efficiency: {aggregated['avg_path_efficiency']:.3f}")
        
        return aggregated
    
    def create_animated_demo(self, model, algorithm, scenario='medium', num_bots=5):
        """Create animated demonstration of navigation"""
        print(f"🎬 Creating animated demo for {algorithm}...")
        
        env = self.create_navigation_env(scenario, num_bots)
        obs, _ = env.reset()
        
        # Setup animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Main plot setup
        scenario_config = self.scenarios[scenario]
        start_point = np.array(scenario_config['start'])
        target_point = np.array(scenario_config['target'])
        
        ax1.set_xlim(min(start_point[0], target_point[0]) - 2, 
                    max(start_point[0], target_point[0]) + 2)
        ax1.set_ylim(min(start_point[1], target_point[1]) - 2, 
                    max(start_point[1], target_point[1]) + 2)
        ax1.set_aspect('equal')
        ax1.set_title(f'{algorithm} Navigation: {scenario.title()} Distance')
        ax1.grid(True, alpha=0.3)
        
        # Plot start and target
        ax1.plot(start_point[0], start_point[1], 'go', markersize=15, label='Start (A)')
        ax1.plot(target_point[0], target_point[1], 'ro', markersize=15, label='Target (B)')
        ax1.plot([start_point[0], target_point[0]], [start_point[1], target_point[1]], 
                'k--', alpha=0.5, label='Direct Path')
        
        # Initialize bot visuals
        bot_circles = []
        speed_labels = []
        trail_lines = []
        colors = plt.cm.Set3(np.linspace(0, 1, num_bots))
        
        for i in range(num_bots):
            circle = Circle((0, 0), 0.2, facecolor=colors[i], 
                          edgecolor='black', alpha=0.8)
            ax1.add_patch(circle)
            bot_circles.append(circle)
            
            label = ax1.text(0, 0, '', fontsize=10, fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor='white', alpha=0.8))
            speed_labels.append(label)
            
            trail, = ax1.plot([], [], '-', color=colors[i], alpha=0.6, linewidth=2)
            trail_lines.append(trail)
        
        ax1.legend()
        
        # Metrics plot setup
        ax2.set_title('Real-time Metrics')
        distance_line, = ax2.plot([], [], 'b-', label='Distance to Target')
        energy_line, = ax2.plot([], [], 'r-', label='Energy Consumption')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Animation data
        trajectory_history = []
        distance_history = []
        energy_history = []
        time_steps = []
        
        def animate(frame):
            nonlocal obs, trajectory_history, distance_history, energy_history
            
            if frame >= env.H:
                return bot_circles + speed_labels + trail_lines + [distance_line, energy_line]
            
            # Get action and step environment
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update positions
            positions = env.state.reshape(num_bots, 2)
            trajectory_history.append(positions.copy())
            
            # Update bot visuals
            for i in range(num_bots):
                pos = positions[i]
                speed = action[i]
                
                # Update bot position
                bot_circles[i].set_center(pos)
                
                # Update speed label
                speed_labels[i].set_position((pos[0], pos[1] + 0.4))
                speed_labels[i].set_text(f'{i+1}\nω={speed:.2f}')
                
                # Update trail
                if len(trajectory_history) > 1:
                    trail_x = [traj[i, 0] for traj in trajectory_history[-20:]]
                    trail_y = [traj[i, 1] for traj in trajectory_history[-20:]]
                    trail_lines[i].set_data(trail_x, trail_y)
            
            # Update metrics
            com = positions.mean(axis=0)
            distance = np.linalg.norm(com - target_point)
            energy = np.sum(action**2)
            
            distance_history.append(distance)
            energy_history.append(energy)
            time_steps.append(frame)
            
            # Update metrics plot
            if len(time_steps) > 1:
                ax2.clear()
                ax2.plot(time_steps, distance_history, 'b-', label='Distance to Target')
                ax2.plot(time_steps, [e/10 for e in energy_history], 'r-', label='Energy/10')
                ax2.set_title(f'Metrics (Step {frame}, Distance: {distance:.2f})')
                ax2.set_xlabel('Time Steps')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            if terminated or truncated:
                success = distance < 0.5
                ax1.set_title(f'{algorithm}: {"SUCCESS" if success else "INCOMPLETE"}')
            
            return bot_circles + speed_labels + trail_lines
        
        # Create and run animation
        anim = animation.FuncAnimation(fig, animate, frames=env.H, 
                                     interval=100, blit=False, repeat=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def run_quick_comparison(self):
        """Run quick comparison of algorithms"""
        print("🚀 Running Quick Navigation Algorithm Comparison")
        print("=" * 60)
        print("Testing RL algorithms on point A → point B navigation")
        print("Using physics from mpc_triangle_alignment.ipynb")
        print()
        
        scenario = 'medium'
        num_bots = 5
        
        results = {}
        
        for algorithm in self.algorithms:
            try:
                print(f"\n🔥 Testing {algorithm}")
                print("-" * 30)
                
                # Create environment
                env = self.create_navigation_env(scenario, num_bots)
                
                # Train algorithm (short training for demo)
                model, training_time = self.train_algorithm(algorithm, env, timesteps=15000)
                
                # Test algorithm
                test_results = self.test_algorithm(model, algorithm, scenario, num_bots, episodes=3)
                test_results['training_time'] = training_time
                
                results[algorithm] = test_results
                
                # Show animated demo for best performing algorithm so far
                if (not results or 
                    test_results['success_rate'] == max([r['success_rate'] for r in results.values()])):
                    print(f"\n🎬 Showing demo for {algorithm} (best so far)...")
                    self.create_animated_demo(model, algorithm, scenario, num_bots)
                
                env.close()
                
            except Exception as e:
                print(f"❌ Error with {algorithm}: {e}")
                results[algorithm] = {'error': str(e)}
        
        # Create comparison visualization
        self.create_comparison_plot(results, scenario)
        
        return results
    
    def create_comparison_plot(self, results, scenario):
        """Create comparison visualization"""
        print("\n📊 Creating comparison visualization...")
        
        # Filter valid results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("❌ No valid results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'🤖 Navigation Algorithm Comparison\n' +
                    f'Scenario: {scenario.title()} Distance (Point A → Point B)', 
                    fontsize=14, fontweight='bold')
        
        algorithms = list(valid_results.keys())
        
        # Success Rate
        success_rates = [valid_results[alg]['success_rate'] for alg in algorithms]
        bars1 = axes[0,0].bar(algorithms, success_rates, color='lightgreen')
        axes[0,0].set_title('🎯 Success Rate')
        axes[0,0].set_ylabel('Success Rate')
        for bar, rate in zip(bars1, success_rates):
            axes[0,0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                          f'{rate:.2%}', ha='center', va='bottom')
        axes[0,0].grid(True, alpha=0.3)
        
        # Path Efficiency
        efficiencies = [valid_results[alg]['avg_path_efficiency'] for alg in algorithms]
        bars2 = axes[0,1].bar(algorithms, efficiencies, color='lightblue')
        axes[0,1].set_title('🛣️ Path Efficiency')
        axes[0,1].set_ylabel('Efficiency Ratio')
        for bar, eff in zip(bars2, efficiencies):
            axes[0,1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                          f'{eff:.3f}', ha='center', va='bottom')
        axes[0,1].grid(True, alpha=0.3)
        
        # Final Distance
        distances = [valid_results[alg]['avg_final_distance'] for alg in algorithms]
        bars3 = axes[1,0].bar(algorithms, distances, color='lightcoral')
        axes[1,0].set_title('📍 Final Distance to Target')
        axes[1,0].set_ylabel('Distance')
        for bar, dist in zip(bars3, distances):
            axes[1,0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                          f'{dist:.2f}', ha='center', va='bottom')
        axes[1,0].grid(True, alpha=0.3)
        
        # Training Time
        training_times = [valid_results[alg]['training_time'] for alg in algorithms]
        bars4 = axes[1,1].bar(algorithms, training_times, color='gold')
        axes[1,1].set_title('⏱️ Training Time')
        axes[1,1].set_ylabel('Time (seconds)')
        for bar, time_val in zip(bars4, training_times):
            axes[1,1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                          f'{time_val:.0f}s', ha='center', va='bottom')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'navigation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comparison plot saved to {self.results_dir / 'navigation_comparison.png'}")
        
        # Print summary
        print("\n🏆 FINAL RESULTS SUMMARY:")
        print("=" * 40)
        for alg in algorithms:
            res = valid_results[alg]
            print(f"{alg}:")
            print(f"  Success Rate: {res['success_rate']:.2%}")
            print(f"  Path Efficiency: {res['avg_path_efficiency']:.3f}")
            print(f"  Final Distance: {res['avg_final_distance']:.2f}")
            print(f"  Training Time: {res['training_time']:.1f}s")
            print()

def main():
    """Main function to run the navigation test"""
    print("🤖 Robotic Swarm Navigation Test System")
    print("=" * 50)
    print("Physics-accurate testing based on MPC triangle alignment")
    print("Tests DDPG, SAC, TD3, PPO on point-to-point navigation")
    print()
    
    # Create test system
    tester = NavigationTestSystem()
    
    # Run comparison
    results = tester.run_quick_comparison()
    
    print("\n🎉 Navigation testing completed!")
    print("Check the 'navigation_results' folder for saved models and plots.")

if __name__ == "__main__":
    main() 