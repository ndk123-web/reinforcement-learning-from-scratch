# Quick Concept & Terminology Reference

A concise summary of all Reinforcement Learning (RL) concepts, parameters, and terms used in this project:

---

## 🧠 Core RL Concepts & Terms

- **Agent**: The learner/decision-maker navigating the grid.
- **Environment**: The $4 \times 4$ GridWorld maze with boundary walls and obstacles.
- **State ($s \in S$)**: The agent's current position (16 states: indices `0` to `15`).
- **Action ($a \in A$)**: The movement choices (`0: Up`, `1: Down`, `2: Left`, `3: Right`).
- **Reward ($r$)**: Feedback signal (+10 for goal, -5 for obstacles, -1 for normal/boundary steps).
- **Q-Table ($Q(s, a)$)**: A lookup table ($16 \times 4$) storing expected cumulative rewards for state-action pairs.
- **Policy ($\pi(s)$)**: The strategy mapping each state to the best action (`argmax Q[s]`).
- **Q-Learning**: A model-free, off-policy Temporal Difference (TD) algorithm to learn the optimal policy.
- **Bellman Equation**: Update rule: $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$.
- **Episode**: A complete run from the start state (`0`) until reaching the terminal goal state (`15`).

---

## ⚙️ Key Hyperparameters Explained

### 1. **Gamma ($\gamma$) — Discount Factor (`0.95`)**
- **Purpose**: Balances **immediate rewards vs. future rewards**.
- **$\gamma \to 0$ (Short-sighted):** Only values the next immediate reward.
- **$\gamma \to 1$ (Far-sighted):** Values long-term cumulative rewards (reaching the goal).

### 2. **Alpha ($\alpha$) — Learning Rate (`0.1`)**
- **Purpose**: Controls **how fast new experience overwrites old knowledge**.
- **$\alpha = 0$:** Learns nothing (Q-values never change).
- **$\alpha = 1$:** Discards previous history, only remembering the latest transition.
- **$\alpha = 0.1$:** Ensures steady, stable convergence.

### 3. **Epsilon ($\epsilon$) — Exploration Rate (`1.0` $\to$ `0.01`)**
- **Purpose**: Balances **Exploration** (discovering unknown paths) vs. **Exploitation** (taking the best known action).
- **$\epsilon = 1.0$ (Training Start):** 100% random actions to explore the maze.
- **$\epsilon \to 0.01$ (Training End via Decay $\times 0.995$):** 99% greedy choices following the learned optimal policy.
