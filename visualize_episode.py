#!/usr/bin/env python3
"""
Visualise one episode of the trained TD3-LSTM policy.

• loads  models/best/best_model.zip   (fallback: models/final.zip)
• runs HydroSwarmEnv for ≤ 400 steps
• shows live positions, trails and target
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from stable_baselines3 import TD3
from src.envs.hydro_swarm_env import HydroSwarmEnv

# ─────────────────────────────────────────────────────────── #
#  1. load env & model
# ─────────────────────────────────────────────────────────── #
env = HydroSwarmEnv()
try:
    model = TD3.load("models/best/best_model.zip", env=env)
except FileNotFoundError:
    model = TD3.load("models/final.zip", env=env)

obs, _ = env.reset()

# ─────────────────────────────────────────────────────────── #
#  2. Matplotlib setup
# ─────────────────────────────────────────────────────────── #
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-8, 8)
ax.set_ylim(-4, 4)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.set_title("Hydro-Swarm Navigation (TD3-LSTM)")

colors = plt.cm.Set3(np.linspace(0, 1, env.N))
bot_circles = [Circle((0, 0), 0.3, color=c, ec="k", lw=1.5) for c in colors]
for c in bot_circles:
    ax.add_patch(c)

# start & goal
ax.plot(env.start[0], env.start[1], "g^", markersize=14, label="Start")
ax.plot(env.goal[0], env.goal[1], "r*", markersize=18, label="Goal")
ax.legend()

trail_hist: list[np.ndarray] = []       # store last 40 positions

# ─────────────────────────────────────────────────────────── #
#  3. run episode
# ─────────────────────────────────────────────────────────── #
MAX_STEPS = 400
for step in range(MAX_STEPS):
    # policy action
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, trunc, info = env.step(action)

    # store trail
    trail_hist.append(np.array(info["positions"]))
    if len(trail_hist) > 40:
        trail_hist.pop(0)

    # update plot
    ax.set_title(f"Step {step:3d} | Dist to goal = {info['distance']:.2f} m")
    for i, circle in enumerate(bot_circles):
        x, y = info["positions"][i]
        circle.center = (x, y)

    # trails
    for bot_idx in range(env.N):
        trail = np.array([pos[bot_idx] for pos in trail_hist])
        ax.plot(trail[:, 0], trail[:, 1], color=colors[bot_idx], alpha=0.4, lw=1)

    plt.pause(0.05)          # ~20 FPS

    if done:
        ax.set_title(f"🎉 SUCCESS in {step} steps")
        break
    if trunc:
        ax.set_title("Episode truncated (time-out)")
        break

plt.ioff()
print("Visualisation complete – close the window to exit.")
plt.show() 