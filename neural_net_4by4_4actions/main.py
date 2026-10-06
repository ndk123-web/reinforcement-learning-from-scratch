import tensorflow as tf
import numpy as np
import random


# ============================================================
# ENVIRONMENT
# ============================================================

ROWS, COLS = 4, 4

START = (0, 0)
GOAL = (3, 3)

OBSTACLES = {
    (1, 2),
    (1, 3),
}

# 0 = UP
# 1 = DOWN
# 2 = LEFT
# 3 = RIGHT
ACTIONS = [0, 1, 2, 3]


# ============================================================
# ENVIRONMENT FUNCTIONS
# ============================================================


def step(state, action):
    """
    Takes current state + action
    and returns:
        next_state
        reward
        done
    """

    row, col = state

    # Calculate next position
    if action == 0:  # UP
        next_state = (row - 1, col)

    elif action == 1:  # DOWN
        next_state = (row + 1, col)

    elif action == 2:  # LEFT
        next_state = (row, col - 1)

    elif action == 3:  # RIGHT
        next_state = (row, col + 1)

    # --------------------------------------------------------
    # Outside the maze
    # --------------------------------------------------------

    if (
        next_state[0] < 0
        or next_state[0] >= ROWS
        or next_state[1] < 0
        or next_state[1] >= COLS
    ):
        return state, -10, True

    # --------------------------------------------------------
    # Obstacle
    # --------------------------------------------------------

    if next_state in OBSTACLES:
        return state, -10, True

    # --------------------------------------------------------
    # Goal
    # --------------------------------------------------------

    if next_state == GOAL:
        return next_state, 10, True

    # --------------------------------------------------------
    # Normal movement
    # --------------------------------------------------------

    return next_state, -1, False


# ============================================================
# MODEL
# ============================================================


def create_model():

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(4),
        ]
    )

    return model


# ============================================================
# STATE CONVERSION
# ============================================================


def state_to_tensor(state):
    """
    Converts:

        (row, col)

    into:

        [[row, col]]
    """

    row, col = state

    return tf.constant([[float(row), float(col)]], dtype=tf.float32)


# ============================================================
# TRAINING
# ============================================================


def train():

    model = create_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    gamma = 0.9

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    episodes = 1000

    for episode in range(episodes):

        state = START

        total_reward = 0

        for step_number in range(50):

            # =================================================
            # EPSILON-GREEDY ACTION
            # =================================================

            if random.random() < epsilon:

                # Explore
                action = random.choice(ACTIONS)

            else:

                # Exploit
                state_tensor = state_to_tensor(state)

                q_values = model(state_tensor)

                action = int(tf.argmax(q_values[0]).numpy())

            # =================================================
            # ENVIRONMENT
            # =================================================

            next_state, reward, done = step(state, action)

            total_reward += reward

            # =================================================
            # CURRENT Q VALUES
            # =================================================

            state_tensor = state_to_tensor(state)

            with tf.GradientTape() as tape:

                q_values = model(state_tensor)
                print("q_values: ", q_values)

                # Q(s,a)
                current_q = q_values[0, action]

                # =================================================
                # BELLMAN TARGET
                # =================================================

                if done:

                    target = tf.constant(float(reward), dtype=tf.float32)

                else:

                    next_state_tensor = state_to_tensor(next_state)

                    next_q_values = model(next_state_tensor)

                    max_next_q = tf.reduce_max(next_q_values[0])

                    target = reward + gamma * max_next_q

                # =================================================
                # LOSS
                # =================================================

                loss = tf.square(current_q - target)

            # =================================================
            # BACKPROPAGATION
            # =================================================

            gradients = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Move to next state
            state = next_state

            if done:
                break

        # =====================================================
        # EPSILON DECAY
        # =====================================================

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress
        if (episode + 1) % 100 == 0:

            print(
                f"Episode: {episode + 1:4d} | "
                f"Reward: {total_reward:5.1f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    return model


# ============================================================
# SHOW Q VALUES
# ============================================================


def show_q_values(model):

    print("\n")
    print("=" * 60)
    print("LEARNED Q VALUES")
    print("=" * 60)

    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    for row in range(ROWS):

        for col in range(COLS):

            state = (row, col)

            # Obstacle
            if state in OBSTACLES:

                print(f"{state}: OBSTACLE")

                continue

            # Goal
            if state == GOAL:

                print(f"{state}: GOAL")

                continue

            state_tensor = state_to_tensor(state)

            q_values = model(state_tensor)[0].numpy()

            best_action = np.argmax(q_values)

            print(
                f"{state} -> "
                f"UP={q_values[0]:7.2f}, "
                f"DOWN={q_values[1]:7.2f}, "
                f"LEFT={q_values[2]:7.2f}, "
                f"RIGHT={q_values[3]:7.2f} "
                f"| BEST={action_names[best_action]}"
            )

        print()


# ============================================================
# TEST THE LEARNED AGENT
# ============================================================


def test(model):

    print("\n")
    print("=" * 60)
    print("TESTING AGENT")
    print("=" * 60)

    state = START

    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    for step_number in range(20):

        state_tensor = state_to_tensor(state)

        q_values = model(state_tensor)[0]

        action = int(tf.argmax(q_values).numpy())

        next_state, reward, done = step(state, action)

        print(
            f"Step {step_number + 1:2d}: "
            f"{state} --"
            f"{action_names[action]}"
            f"--> {next_state} "
            f"| reward={reward}"
        )

        state = next_state

        if done:

            if state == GOAL:
                print("\nGOAL REACHED!")

            else:
                print("\nEpisode ended.")

            break


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    model = train()

    show_q_values(model)

    test(model)
