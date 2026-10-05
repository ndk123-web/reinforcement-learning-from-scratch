# Quick RL Concept & Hyperparameter Cheat Sheet

A concise reference of all Reinforcement Learning (RL) terminology and parameters across this repository.

---

## 📌 Core RL Concepts

- **Agent**: The learner that takes actions in the maze environment.
- **Environment**: GridWorld matrices ($4 \times 4$ and $5 \times 5$) with obstacles and boundaries.
- **State ($s \in S$)**: The agent's current position cell index (`0` to `ROWS * COLS - 1`).
- **Action ($a \in A$)**: Directional movement choices (Orthogonal: `4 actions`, or Diagonal-inclusive: `6 actions`).
- **Reward ($r$)**: Scalar environmental feedback (+10 for goal, -5 for obstacles, -1 for regular/wall steps).
- **Q-Table ($Q(s, a)$)**: Lookup matrix storing expected cumulative returns for each state-action pair.
- **Policy ($\pi(s)$)**: Decision rule picking the best action from learned values ($\arg\max_a Q(s, a)$).
- **Bellman Equation**: TD update formula:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
- **Episode / Epoch**: One full trajectory starting from `START` state until reaching `GOAL` or terminating.

---

## ⚙️ Hyperparameters Explained

### 1. **Gamma ($\gamma$) — Discount Factor**
- **Definition**: Determines how much the agent values future rewards compared to immediate rewards.
- **$\gamma \to 0$ (Short-sighted):** Cares only about the next step reward.
- **$\gamma \to 1$ (Far-sighted):** Highly values reaching the long-term goal.

### 2. **Alpha ($\alpha$) — Learning Rate**
- **Definition**: Controls the rate at which newly acquired information overwrites old Q-values.
- **$\alpha = 0$:** No learning occurs.
- **$\alpha = 1$:** Discards previous history, relying entirely on the newest step transition.
- **Low $\alpha$ (e.g., $0.01$ - $0.1$):** Ensures stable, gradual convergence.

### 3. **Epsilon ($\epsilon$) — Exploration Rate**
- **Definition**: Governs the $\epsilon$-greedy balance between **Exploration** (discovering unknown paths) and **Exploitation** (choosing the highest Q-value).
- **$\epsilon = 1.0$ (Start):** 100% random actions to explore the maze topology.
- **$\epsilon \to 0.01$ (Decay):** Gradually shifts to exploiting the learned optimal policy.
