import numpy as np
import random

rows, cols = 4, 4
start, goal = 0, 15

# walls on location
obstacles = {5, 9}

# up,down,left,right
actions = [0, 1, 2, 3]
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up  # Down  # Left  # Right

# 16 * 4
Q = np.zeros((rows * cols, len(actions)))

# learning rate
alpha = 0.1

# how much agent values the future reward
gamma = 0.95

# explore vs exploit
epsilon = 1.0

# epochs
episodes = 100


def step(state, action):
    r, c = divmod(state, cols)
    dr, dc = moves[action]
    nr, nc = r + dr, c + dc

    # invalid movements
    if not (0 <= nr < rows and 0 <= nc < cols):
        return state, -1, False

    # convert matrix to actual index
    next_state = nr * cols + nc

    # if it obstacles
    if next_state in obstacles:
        return next_state, -5, False

    if next_state == goal:
        return next_state, 10, True

    return next_state, -1, False


def train():
    global epsilon

    for episode in range(episodes):
        state = start
        done = False

        while not done:

            # Exploitation vs Exploration
            if random.random() < epsilon:
                # explore
                action = random.choice(actions)
            else:
                # exploit (meaning choose that action which has most Q value)
                action = np.argmax(Q[state])

            next_state, reward, done = step(state, action)

            # update Q table
            target = reward

            # if not done then reduce thr reward
            if not done:
                target = target + (gamma * np.max(Q[next_state]))

            # update Q value of particular (state, action)
            Q[state, action] += alpha * (target - Q[state, action])

            state = next_state

        # gradually reduce exploration
        epsilon = max(0.01, epsilon * 0.995)


if __name__ == "__main__":
    train()
    arrows = ["↑", "↓", "←", "→"]

    for s in range(rows * cols):
        if s == goal:
            print("G", end=" ")
        elif s in obstacles:
            print("#", end=" ")
        else:
            print(arrows[np.argmax(Q[s])], end=" ")
        if s % cols == cols - 1:
            print()
        
    np.save("q_table", Q) 

print("Q table: \n", np.round(Q, 2))
