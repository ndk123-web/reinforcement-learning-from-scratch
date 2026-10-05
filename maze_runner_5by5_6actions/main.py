import random
import numpy as np

# Enviroment
ACTIONS = [0, 1, 2, 3, 4, 5]
MOVES = [(0, 1), (1, 0), (1, 1), (-1, -1), (0, -1), (-1, 0)]
OBSTACLES = {8, 12, 13, 14, 18}
ROWS, COLS = 5, 5
START, GOAL = 0, 24

# Rules
ALPHA = 0.01
GAMMA = 0.92
EPSILON = 1.0
EPOCHS = 1000

Q = np.zeros((ROWS * COLS, len(ACTIONS)))


def step(state, action):
    r, c = divmod(state, COLS)
    dr, dc = MOVES[action]
    nr, nc = r + dr, c + dc

    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        return state, -1, False

    next_state = nr * COLS + nc

    if next_state == GOAL:
        return next_state, 10, True

    if next_state in OBSTACLES:
        return next_state, -5, False

    return next_state, -1, False


def train():
    global EPSILON

    for epoch in range(EPOCHS):
        state = START
        done = False

        while not done:

            # select action either (explore or exploit)
            if random.random() < EPSILON:
                action = random.choice(ACTIONS)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = step(state, action)

            target = reward

            if not done:
                target = target + (GAMMA * np.max(Q[next_state]))

            Q[state, action] += ALPHA * (target - Q[state, action])

            state = next_state

        EPSILON = max(0.01, EPSILON * 0.995)


if __name__ == "__main__":
    train()

    print("Q:")
    print(np.round(Q, 2))

    MOVES = ["R", "D", "RD", "LD", "L", "U"]

    print("\nPolicy Map:")
    for s in range(ROWS * COLS):
        if s == GOAL:
            symbol = "G"
        elif s in OBSTACLES:
            symbol = "##"
        else:
            symbol = MOVES[np.argmax(Q[s])]

        print(f"{symbol:>2}", end=" ")

        if s % COLS == COLS - 1:
            print()
  

