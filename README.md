# Reinforcement Learning from Scratch

A clean, modular repository implementing tabular **Q-Learning** from scratch in Python to solve discrete GridWorld maze navigation problems.

---

## 📂 Project Structure

```
reinforcement-learning/
├── maze_runner_4by4_4actions/     # Classic 4x4 maze with 4 orthogonal actions (U, D, L, R)
│   ├── main.py
│   └── ...
├── maze_runner_5by5_6actions/     # 5x5 maze with 6 actions (including diagonal moves: RD, LD)
│   ├── main.py
│   └── README.md
├── README.md                      # Repository overview & deep-dive
└── Short_Readme.md                # Quick RL concept & parameter cheat sheet
```

---

## 🔬 Environments & Implementations

### 1. [4x4 Maze (4 Actions)](file:///c:/Users/Navnath/OneDrive/Desktop/nlp/reinforcement-learning/maze_runner_4by4_4actions/main.py)
- **Grid Size**: $4 \times 4$ (16 states: `0` to `15`)
- **Actions**: `4` (Up, Down, Left, Right)
- **Obstacles**: States `{5, 9}`
- **Goal**: State `15`
- **Episodes**: `100`

### 2. [5x5 Maze (6 Actions)](file:///c:/Users/Navnath/OneDrive/Desktop/nlp/reinforcement-learning/maze_runner_5by5_6actions/main.py)
- **Grid Size**: $5 \times 5$ (25 states: `0` to `24`)
- **Actions**: `6` (Right, Down, Right-Down, Left-Down, Left, Up)
- **Obstacles**: Wall block states `{8, 12, 13, 14, 18}`
- **Goal**: State `24`
- **Episodes**: `1000`

---

## 🧠 Q-Learning Mechanics

The agent updates its state-action value table $Q(s, a)$ using the Temporal Difference Bellman update rule:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

- **$\epsilon$-Greedy Policy**: Balances exploration (random movement) and exploitation (highest Q-value action) with exponential decay.
- **Reward Function**:
  - Reaching Goal: `+10` (terminal)
  - Collision with Obstacles: `-5`
  - Normal Step / Boundary Bounce: `-1`

---

## 🚀 Quickstart

### Prerequisites
```bash
pip install numpy
```

### Run 4x4 Maze Solver
```bash
python maze_runner_4by4_4actions/main.py
```

### Run 5x5 Maze Solver
```bash
python maze_runner_5by5_6actions/main.py
```
