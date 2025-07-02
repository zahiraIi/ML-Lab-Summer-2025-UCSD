# Hydrodynamic-Swarm RL Lab  
UCSD Summer 2025

A mini-research sandbox for teaching and testing decentralised control of small underwater "spinning-bot" swarms.

*   **Physics** – exactly the model in `referencefiles/multibot_cluster_env.py` (no edits).  
*   **Environment** – wraps that physics, adds a start-and-goal task.  
*   **RL agent** – TD3 with a tiny LSTM so each robot can handle limited sensing.  
*   **Training script** – one file, no configuration frameworks.

---

## 1  Folder overview

| Path | What's inside |
|------|---------------|
| `referencefiles/` | PDFs, notebooks, and the original `multibot_cluster_env.py` physics. **Do not edit.** |
| `src/envs/hydro_swarm_env.py` | Turns the physics into a Gymnasium environment (adds goal, reward, sonar, etc.). |
| `src/policies/lstm_td3.py` | TD3 actor-critic; the actor passes through an LSTM. |
| `src/training/train.py` | Simple training loop that saves models and evaluation logs. |
| `requirements.txt` | Python packages (torch, stable-baselines 3, …). |

All other directories (`models/`, `logs/`, `tb/`) are created at runtime.

---

## 2  Installation (5 min)

```bash
# 2.1  create and activate a virtual env
python -m venv .venv
source .venv/bin/activate          # Windows →  .venv\Scripts\activate

# 2.2  install dependencies
pip install -r requirements.txt

# 2.3  make sure Python sees the project
touch src/__init__.py src/envs/__init__.py src/policies/__init__.py
```

If you have a GPU with CUDA, PyTorch will pick it up automatically.  
Otherwise everything runs on CPU (just slower).

---

## 3  Training

```bash
# run as a module so Python adds project root to sys.path
python -m src.training.train
```

*   Runs for **500 k steps** (about 20 – 30 min on a laptop CPU).  
*   Writes TensorBoard logs to `tb/` – start TensorBoard in another terminal:  

    ```bash
    tensorboard --logdir tb
    ```

*   Saves best model to `models/best/best_model.zip` and the final model to `models/final.zip`.

---

## 4  What's in the observation & action?

| Item | Description | Size |
|------|-------------|------|
| own position (x, y) | metres | 2 |
| own velocity (vx, vy) | m/s   | 2 |
| vector to goal (dx, dy) | metres | 2 |
| sonar array | 24 radial distances (neighbours only) | 24 |
| neighbour snapshot | for up to 4 neighbours: rel-pos (2) + rel-vel (2) | 16 |
| **total** | | **46** |

Action is a single float **ω ∈ [–2, 2]** – the spinning frequency applied to *all* robots (shared policy).

---

## 5  Running a trained model

```python
from stable_baselines3 import TD3
from src.envs.hydro_swarm_env import HydroSwarmEnv

env   = HydroSwarmEnv()
model = TD3.load("models/best/best_model.zip", env=env)

obs, _ = env.reset()
done   = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    # env.render() is empty – add your own plotting if you want visuals
print("distance to goal:", info["distance"])
```

---

## 6  Extending the lab

* **Visualisation** – add a Matplotlib or PyVista animation in `env.render()`.  
* **Currents** – right now the water is still; introduce a slow drift and see if the policy adapts.  
* **Bigger swarms** – edit `train.py` (`CFG["n_bots"]`) and retrain.  
* **Other algorithms** – swap TD3 for SAC or PPO (change two lines in `train.py`).  
* **Real-world transfer** – randomise physics parameters each episode to make the policy more robust.

---

## 7  Troubleshooting

| Problem | Quick fix |
|---------|-----------|
| `ModuleNotFoundError: src` | Run scripts with `python -m src.…` **or** `export PYTHONPATH=$PWD`. |
| GPU not used | `torch.cuda.is_available()` must be `True`; check CUDA install. |
| Training slow | Reduce `total_steps` in `train.py` (less accurate) or use a GPU. |

---

## 8  Credits

* Base physics from UCSD course materials (`multibot_cluster_env.py`).  
* TD3 idea borrowed from the open-source "[DRL-robot-navigation]" project.  
* Code organised and simplified by the ML-Lab Summer 2025 teaching team.

Pull requests and questions welcome – open an issue or ask during office hours.