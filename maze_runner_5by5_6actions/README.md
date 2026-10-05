# 5x5 Grid Maze Runner with 6 Actions (Q-Learning)

An extended tabular **Q-Learning** implementation on a $5 \times 5$ GridWorld maze that introduces **diagonal movement actions** alongside standard orthogonal movements.

---

## 📌 Environment Setup

The agent navigates a $5 \times 5$ discrete grid (25 states: `0` to `24`) with complex obstacle configurations:

```text
+----+----+----+----+----+
| S  |  1 |  2 |  3 |  4 |    S  : Start (State 0)
+----+----+----+----+----+
|  5 |  6 |  7 | ## |  9 |    ## : Obstacles (States 8, 12, 13, 14, 18)
+----+----+----+----+----+
| 10 | 11 | ## | ## | ## |    G  : Goal (State 24)
+----+----+----+----+----+
| 15 | 16 | 17 | ## | 19 |
+----+----+----+----+----+
| 20 | 21 | 22 | 23 | G  |
+----+----+----+----+----+
```

---

## 🕹️ Action Space (6 Actions)

Unlike the standard 4-directional grid, this environment allows **6 discrete actions** including diagonals:

| Action ID | Name | Direction | Vector `(dr, dc)` |
| :---: | :---: | :---: | :---: |
| `0` | **R** | Right | `(0, 1)` |
| `1` | **D** | Down | `(1, 0)` |
| `2` | **RD** | Right-Down (Diagonal) | `(1, 1)` |
| `3` | **LD** | Left-Down (Diagonal) | `(-1, -1)` / `(1, -1)` |
| `4` | **L** | Left | `(0, -1)` |
| `5` | **U** | Up | `(-1, 0)` |

---

## 🎯 Reward Structure

| Event | Reward | Terminal |
| :--- | :---: | :---: |
| **Reach Goal** (`State 24`) | `+10` | Yes |
| **Hit Obstacle** (`States 8, 12, 13, 14, 18`) | `-5` | No |
| **Step Penalty / Wall Collision** | `-1` | No |

---

## 🎛️ Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Learning Rate ($\alpha$)** | `0.01` | Conservative step size for larger state-action space |
| **Discount Factor ($\gamma$)** | `0.92` | Balances future vs. immediate reward |
| **Initial Epsilon ($\epsilon$)** | `1.0` | Starts with 100% random exploration |
| **Epsilon Decay** | `0.995` | Decays down to minimum `0.01` |
| **Epochs / Episodes** | `1000` | Sufficient iterations to discover diagonal optimal shortcuts |

---

## 🚀 Running the Script

```bash
cd maze_runner_5by5_6actions
python main.py
```
