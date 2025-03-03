import numpy as np
import matplotlib.pyplot as plt
import random


class WindyGridworld:
    """WindyGridworld defines an environment for an agent according
    to Sutton & Barto pg. 130"""

    def __init__(self):
        # Grid size 7 x 10
        self.rows = 7
        self.cols = 10

        # Reward always -1
        self.reward = -1

        # Start, goal positions
        self.start_state = (3, 0)
        self.goal_state = (3, 7)

        # Current position
        self.state = self.start_state

        # Upward wind strength for each column
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        # Actions: up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.num_actions = len(self.actions)

    def reset(self):
        # Reset to start state
        self.state = self.start_state
        return self.start_state

    def step(self, action):
        """step function takes an action and applies it to the agent,
        given wind and that the agent cannot go outside the gridworld."""

        row, col = self.state

        row_move, col_move = self.actions[action]

        next_row = row + row_move
        next_col = col + col_move

        # Account for wind effect
        next_row = next_row - self.wind[col]

        # Stay within the gridworld
        next_row = max(0, min(next_row, self.rows - 1))
        next_col = max(0, min(next_col, self.cols - 1))

        next_state = (next_row, next_col)

        # Check if goal state reached
        if next_state == self.goal_state:
            done = True
        else:
            done = False

        self.state = next_state
        return next_state, self.reward, done


def epsilon_greedy_policy(Q, state, epsilon, num_actions):
    if random.uniform(0, 1) < epsilon:
        # Random action
        return random.randint(0, num_actions - 1)
    else:
        # Greedy action
        return np.argmax(Q[state[0], state[1]])


def sarsa(env, episodes, alpha, gamma, epsilon):
    """Use Sarsa control algorithm for given parameter values."""
    # Q hold all values for (row, col, action) triplets
    Q = np.zeros((env.rows, env.cols, env.num_actions))

    # Store total rewards and steps per episode for plotting
    total_rewards = np.zeros(episodes)
    steps_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        # Reset gridworld environment
        state = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        # Choose initial action using epsilon-greedy
        action = epsilon_greedy_policy(Q, state, epsilon, env.num_actions)

        # Continue until reaching the goal
        while not done:
            # Take action, observe new state and reward
            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1

            # Choose next action with epsilon-greedy policy
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.num_actions)

            # Sarsa update rule
            Q[state[0], state[1], action] += alpha * (
                reward
                + gamma * Q[next_state[0], next_state[1], next_action]
                - Q[state[0], state[1], action]
            )

            # Move to next time step
            state = next_state
            action = next_action

        # Record metrics for this episode
        total_rewards[episode] = total_reward
        steps_per_episode[episode] = step_count

    return Q, total_rewards, steps_per_episode


def visualize_policy(env, Q):
    # Create a grid to visualize the policy
    policy = np.argmax(Q, axis=2)

    # Create a 2D grid of characters representing the policy
    policy_grid = np.full((env.rows, env.cols), " ", dtype=object)

    for row in range(env.rows):
        for col in range(env.cols):
            if (row, col) == env.goal_state:
                policy_grid[row, col] = "G"
            # elif (row, col) == env.start_state:
            #     policy_grid[row, col] = "S"
            else:
                action = policy[row, col]
                if action == 0:  # up
                    policy_grid[row, col] = "↑"
                elif action == 1:  # down
                    policy_grid[row, col] = "↓"
                elif action == 2:  # left
                    policy_grid[row, col] = "←"
                elif action == 3:  # right
                    policy_grid[row, col] = "→"

    # Mark wind strengths at the bottom
    wind_str = ""
    for w in env.wind:
        wind_str += f" {w} "

    # Print the policy grid
    print("Policy Grid (S=Start, G=Goal):")
    for row in range(env.rows):
        print("|", end="")
        for col in range(env.cols):
            print(f" {policy_grid[row, col]} |", end="")
        print()

    print("\nWind strength per column:")
    print(wind_str)


def plot_learning_curve(total_rewards, steps_per_episode):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the total reward per episode
    ax1.plot(total_rewards)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Total Reward per Episode")

    # Plot the steps per episode
    ax2.plot(steps_per_episode)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Steps per Episode")

    # Apply a smoothing window to make trends more visible
    window_size = 10
    smoothed_steps = np.convolve(
        steps_per_episode, np.ones(window_size) / window_size, mode="valid"
    )
    ax2.plot(
        range(window_size - 1, len(steps_per_episode)),
        smoothed_steps,
        "r-",
        alpha=0.5,
        label=f"Smoothed (window={window_size})",
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create environment
    env = WindyGridworld()

    # Train SARSA agent
    print("Training SARSA agent...")
    Q, total_rewards, steps = sarsa(
        env, episodes=170, alpha=0.5, gamma=1.0, epsilon=0.1
    )

    # Visualize the learned policy
    visualize_policy(env, Q)

    # Plot learning curves
    plot_learning_curve(total_rewards, steps)

    # Print final statistics
    print(f"\nFinal performance (averaged over last 10 episodes):")
    print(f"Average steps: {np.mean(steps[-10:]):.2f}")
    print(f"Average reward: {np.mean(total_rewards[-10:]):.2f}")
