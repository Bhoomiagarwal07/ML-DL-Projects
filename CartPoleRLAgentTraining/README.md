# Cart-Pole Reinforcement Learning Agent Training (DQN)

## 📌 Objective
Train a reinforcement learning agent to balance a pole on a moving cart — the classic
**CartPole** control problem — using **Deep Q-Networks (DQN)**.

## 🎮 Environment
**CartPole-v1** (Gymnasium)
Docs: [https://gymnasium.farama.org/environments/classic_control/cart_pole/](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

- **Observation space:** 4 continuous values (cart position, cart velocity, pole angle, pole angular velocity)
- **Action space:** 2 discrete actions (push left / push right)
- **Reward:** +1 per timestep the pole stays upright, up to a 500-step episode cap

## 🛠️ Libraries Used
- `gymnasium` — the RL environment
- `stable-baselines3` — DQN implementation, evaluation utilities
- `numpy` / `matplotlib` — numerical operations and visualization

## 🔍 Methodology
1. **Environment Understanding** — explored the observation/action spaces and established an
   untrained random-agent baseline for comparison.
2. **Model Development** — built a DQN agent (2-layer, 256-neuron network) with tuned
   hyperparameters (experience replay, target network, epsilon-greedy exploration), trained
   for 50,000 timesteps.
3. **Evaluation** — evaluated the trained agent over 20 episodes, compared against the random
   baseline, and visualized pole angle/cart position traces during a sample episode.

## 📈 Results

| Agent | Result |
|-------|--------|
| Random (untrained) | 38 steps (single episode) |
| **Trained DQN** | **500.0 ± 0.0** (mean over 20 episodes — the maximum possible score) |

The trained agent perfectly solves CartPole, achieving the maximum reward with zero variance
across all 20 evaluation episodes — a ~13x improvement over the random baseline.

## ✅ Conclusion
This project trained a Deep Q-Network (DQN) agent to solve the classic CartPole balancing
task using Stable-Baselines3. After training for 50,000 timesteps with a tuned hyperparameter
configuration (a 2-layer, 256-neuron network, frequent gradient updates, and fast target
network synchronization), the agent achieved a perfect mean evaluation reward of 500.0 ± 0.0
across 20 episodes — the maximum possible score — compared to just a handful of steps for an
untrained random-action baseline. This demonstrates DQN's ability to learn an effective
control policy purely from trial-and-error interaction with the environment, using experience
replay and a target network to stabilize training. A key characteristic of DQN observed
during this project is its sensitivity to hyperparameter choices: less carefully tuned
settings converged to a noticeably weaker, partially-balancing policy, highlighting a broader
limitation of value-based deep RL methods — they often require careful tuning of learning
rate, network architecture, exploration schedule, and update frequency to reliably reach
optimal performance, and can be less sample-efficient or stable than some alternative RL
algorithms (such as PPO) on certain problems.

## 📂 Files
- `CartPoleRLAgentTraining.ipynb` — full notebook with environment setup, DQN training, and evaluation
