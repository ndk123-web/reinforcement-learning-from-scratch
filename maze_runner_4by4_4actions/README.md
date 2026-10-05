# Reinforcement Learning from Scratch: 4x4 Maze Solver

A clean, dependency-minimal implementation of tabular **Q-Learning** in Python to solve a 4x4 grid maze with obstacles from scratch.

---

## 📌 Overview

This project implements an autonomous agent trained using standard **Q-Learning (Model-Free, Off-Policy RL)** to find the optimal path from a starting position to a target goal while avoiding obstacles and boundaries.

```
+---+---+---+---+
| S | . | . | . |    S : Start (State 0)
+---+---+---+---+
| . | # | . | . |    # : Obstacles (States 5, 9)
+---+---+---+---+
| . | # | . | . |    G : Goal (State 15)
+---+---+---+---+
| . | . | . | G |
+---+---+---+---+
```

---

## ⚙️ How It Works

### 1. State & Action Space
- **State Space**: 16 discrete states ($4 \times 4$ grid, indexed `0` to `15`).
- **Action Space**: 4 discrete actions:
  - `0`: Up (`↑`, `[-1, 0]`)
  - `1`: Down (`↓`, `[1, 0]`)
  - `2`: Left (`←`, `[0, -1]`)
  - `3`: Right (`→`, `[0, 1]`)

### 2. Reward Structure
| Condition | Reward | Terminal |
| :--- | :--- | :--- |
| **Reach Goal** (`State 15`) | `+10` | Yes |
| **Hit Obstacle** (`States 5, 9`) | `-5` | No |
| **Hit Grid Boundary** | `-1` (agent stays in place) | No |
| **Regular Step** | `-1` | No |

### 3. Q-Learning Algorithm
The agent updates its action-value function using the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

- **Exploration vs Exploitation**: $\epsilon$-greedy strategy with exponential decay ($Epsilon \times 0.995$ per episode).

---

## 🎛️ Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate ($\alpha$)** | `0.1` | Step size for Q-value updates |
| **Discount Factor ($\gamma$)** | `0.95` | Importance of future rewards |
| **Initial Epsilon ($\epsilon$)** | `1.0` | Initial exploration probability |
| **Min Epsilon** | `0.01` | Minimum exploration threshold |
| **Episodes** | `100` | Total training iterations |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- NumPy

### Installation
```bash
pip install numpy
```

### Run Training & Evaluation
```bash
python main.py
```

---

## 📊 Output

Upon completion, the script prints:
1. **Learned Policy Map**: Arrow directions representing the best action for each state (`↑`, `↓`, `←`, `→`), `#` for obstacles, and `G` for the goal.
2. **Final Q-Table**: The full $16 \times 4$ value matrix.

Example Policy Grid Output:
```text
→ → ↓ ← 
↓ # ↓ ↓ 
↓ # ↓ ↓ 
→ → → G 
```
