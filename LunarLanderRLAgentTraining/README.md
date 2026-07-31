# Lunar Lander Reinforcement Learning Agent Training (PPO)

## 📌 Objective
Train a reinforcement learning agent to safely land a spacecraft on the moon's surface — the
classic **Lunar Lander** control problem — using **Proximal Policy Optimization (PPO)**.

## 🎮 Environment
**LunarLander-v3** (Gymnasium, requires Box2D)
Docs: [https://gymnasium.farama.org/environments/box2d/lunar_lander/](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

- **Observation space:** 8 continuous values (position, velocity, angle, angular velocity, leg contacts)
- **Action space:** 4 discrete actions (do nothing, fire left/main/right engine)
- **Reward:** landing bonus/crash penalty + shaping rewards + fuel cost
- **"Solved" threshold:** average reward ≥ 200 over 100 consecutive episodes

## 🛠️ Libraries Used
- `gymnasium[box2d]` — the RL environment and physics engine
- `stable-baselines3` — PPO implementation, evaluation utilities, vectorized environments
- `numpy` / `matplotlib` — numerical operations and visualization

## 🔍 Methodology
1. **Environment Understanding** — explored the observation/action spaces and established an
   untrained random-agent baseline (which crashes almost immediately).
2. **Model Development** — trained a PPO agent using 8 parallel environments and tuned
   hyperparameters (`gamma=0.999`, `gae_lambda=0.98`, `ent_coef=0.01`) for 600,000 timesteps.
3. **Evaluation** — evaluated the trained agent over 20 episodes, compared against the random
   baseline, and visualized the lander's trajectory and orientation during a sample landing.

## 📈 Results

| Agent | Result |
|-------|--------|
| Random (untrained) | Strongly negative reward (crashes almost immediately) |
| **Trained PPO** | **229.9 ± 14.8** (mean over 20 episodes) |

The trained agent clears LunarLander's standard "solved" threshold of 200, demonstrating a
genuinely learned, stable landing strategy.

## ✅ Conclusion
This project trained a Proximal Policy Optimization (PPO) agent to solve the Lunar Lander
control task using Stable-Baselines3, achieving a mean evaluation reward of 229.9 ± 14.8 over
20 episodes — clearing the standard "solved" threshold of 200 — compared to a strongly
negative reward for an untrained random-action baseline. Reaching this result required
substantially more training (600,000 timesteps across 8 parallel environments) than the
simpler CartPole task, reflecting LunarLander's more complex, continuous-physics control
problem with sparse, delayed landing rewards and ongoing fuel costs. PPO's clipped policy
update mechanism, which limits how much the policy can change in a single update step, proved
well-suited to this harder problem, providing more stable training than a naive policy
gradient approach might achieve. A key limitation observed in this project is the substantial
compute and wall-clock time required to reach a strong policy on a problem of this
complexity — even with 8 parallel environments, training took several minutes, and further
improvements (such as more consistent, lower-variance landings) would likely require
additional timesteps, hyperparameter tuning, or techniques like reward shaping or curriculum
learning to accelerate convergence further.

## 📂 Files
- `LunarLanderRLAgentTraining.ipynb` — full notebook with environment setup, PPO training, and evaluation
