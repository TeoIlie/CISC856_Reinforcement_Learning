import random
import numpy as np


class WindyGridworld:
    def __init__(self, width, height, start, goal, wind):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.state = start
        self.wind = wind

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state
        if action == 0:  # Up
            row = max(row - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.height - 1)
        elif action == 2:  # Left
            col = max(col - 1, 0)
        elif action == 3:  # Right
            col = min(col + 1, self.width - 1)

        next_state = (row, col)

        if next_state == self.goal:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        self.state = next_state
        return next_state, reward, done


def sarsa(env, episodes, alpha, gamma, epsilon):
    # Initialize Q-table with zeros
    Q = np.zeros((env.height, env.width, 4))

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            # SARSA update rule
            Q[state[0], state[1], action] += alpha * (
                reward
                + gamma * Q[next_state[0], next_state[1], next_action]
                - Q[state[0], state[1], action]
            )

            state = next_state
            action = next_action

    return Q


def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Random action
    else:
        return np.argmax(Q[state[0], state[1]])  # Greedy action


if __name__ == "__main__":
    # Define the grid world environment
    width = 10
    height = 7
    start = (3, 0)
    goal = (3, 7)
    wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

    env = WindyGridworld(width, height, start, goal, wind)

    # SARSA parameters
    episodes = 8000
    alpha = 0.5  # Learning rate
    gamma = 1.0  # Discount factor
    epsilon = 0.1  # Exploration rate

    # Run SARSA
    Q = sarsa(env, episodes, alpha, gamma, epsilon)

    # Print the learned Q-values
    print("Learned Q-values:")
    print(Q)
