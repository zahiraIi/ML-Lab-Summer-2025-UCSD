#!/usr/bin/env python3
"""
TD3-LSTM training script (no Hydra).

Usage
-----
$ python src/training/train.py
"""

from __future__ import annotations
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from src.envs.hydro_swarm_env import HydroSwarmEnv
from src.policies.lstm_td3 import LSTMTD3Policy

# ────────────────────────────
# CONFIG  – edit here only
# ────────────────────────────
CFG = {
    "n_bots": 5,
    "dt": 0.05,
    "T": 25.0,                # episode seconds
    "max_current": 0.0,       # still water (reference physics only)
    #
    "lr": 3e-4,
    "batch": 256,
    "buffer": 200_000,
    "total_steps": 500_000,   # 5 × 10^5
    "eval_freq": 50_000,
}
# ────────────────────────────


def main() -> None:
    env = HydroSwarmEnv(
        n_robots=CFG["n_bots"],
        dt=CFG["dt"],
        episode_seconds=CFG["T"],
    )
    eval_env = HydroSwarmEnv(
        n_robots=CFG["n_bots"],
        dt=CFG["dt"],
        episode_seconds=CFG["T"],
    )

    model = TD3(
        policy=LSTMTD3Policy,
        env=env,
        learning_rate=CFG["lr"],
        batch_size=CFG["batch"],
        buffer_size=CFG["buffer"],
        verbose=1,
        tensorboard_log="tb",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best",
        log_path="logs",
        eval_freq=CFG["eval_freq"],
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=CFG["total_steps"], callback=eval_cb)
    model.save("models/final")
    print("✅ Training finished; model saved to models/final.zip")


if __name__ == "__main__":
    main() 