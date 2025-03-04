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

        # Keep values in gridworld bounds
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

    def get_optimal_path_size(self, Q):
        """Get the length of the optimal path with respect to the current policy π
        that is greedy with respect to Q"""
        # Start from the start state
        current_state = self.reset()
        path_length = 0
        max_path_size = 1000

        while path_length < max_path_size:
            best_action = np.argmax(Q[current_state[0], current_state[1], :])

            current_state, r, done = self.step(best_action)
            path_length += 1

            if done:
                break

        if path_length == max_path_size:
            return f"Optimal path length greater than {max_path_size}"
        else:
            return str(path_length)


def epsilon_greedy_policy(Q, state, epsilon, num_actions):
    """Choose non-greedy action randomly with epislon prob, and
    greedy action the rest of the time"""
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

    # Keep track of stats for graphing
    time_steps = []
    episode_numbers = []
    total_steps = 0
    steps = np.zeros(episodes)

    # Loop for each episode
    for episode in range(episodes):
        # Initialize state and other params
        state = env.reset()
        total_reward = 0
        step_count = 0
        done = False

        # Choose initial action using epsilon-greedy selection
        action = epsilon_greedy_policy(Q, state, epsilon, env.num_actions)

        # Continue until terminal (goal) state reached
        while not done:
            # Take an action, observe reward and new state
            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1
            total_steps += 1

            time_steps.append(total_steps)
            episode_numbers.append(episode)

            # Choose next action with epsilon-greedy policy
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.num_actions)

            # Sarsa update rule for Q function
            Q[state[0], state[1], action] += alpha * (
                reward
                + gamma * Q[next_state[0], next_state[1], next_action]
                - Q[state[0], state[1], action]
            )

            # Move to next state, next action before next time step
            state = next_state
            action = next_action

        steps[episode] = step_count

    return Q, steps, time_steps, episode_numbers


def visualize_policy(env, Q):
    """Creates a vizualization of the policy that is greedy w.r.t Q"""
    # Extract the greedy policy using argmax on the actions dimension at index 2
    policy = np.argmax(Q, axis=2)
    policy_grid = np.full((env.rows, env.cols), "", dtype=object)

    for row in range(env.rows):
        for col in range(env.cols):
            if (row, col) == env.goal_state:
                # If goal or start state, mark separately
                policy_grid[row, col] = "G"
            # elif (row, col) == env.start_state:
            #     policy_grid[row, col] = "S"
            else:
                # Mark greedy actions with arrows
                action = policy[row, col]
                if action == 0:  # up
                    policy_grid[row, col] = "↑"
                elif action == 1:  # down
                    policy_grid[row, col] = "↓"
                elif action == 2:  # left
                    policy_grid[row, col] = "←"
                elif action == 3:  # right
                    policy_grid[row, col] = "→"

    print("Greedy policy π, and wind values at bottom")
    for row in range(env.rows):
        print("| ", end="")
        for col in range(env.cols):
            print(policy_grid[row, col], "| ", end="")
        print()

    wind = ""
    for element in env.wind:
        wind += f"  {str(element)} "
    print(wind + "\n")


def plot_episode_vs_timesteps(time_steps, episode_numbers):
    """Plot the episodes vs timestep to show improvemen over time"""
    plt.figure(figsize=(10, 7))
    plt.scatter(time_steps, episode_numbers, s=3)
    plt.ylabel("Episodes")
    plt.xlabel("Time steps")
    plt.title("Windy Gridworld Learning")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    # Create environment
    env = WindyGridworld()
    num_episodes = 170

    # Train agent with Sarsa
    Q, steps, time_steps, episode_numbers = sarsa(
        env, episodes=num_episodes, alpha=0.5, gamma=1.0, epsilon=0.1
    )

    # Visualize the learned policy
    visualize_policy(env, Q)

    optimal_path_size = env.get_optimal_path_size(Q)
    print("Optimal path length:\n", optimal_path_size)

    average_of_episodes = num_episodes // 2
    print(
        f"Average steps over last {average_of_episodes} episodes:\n {np.mean(steps[-num_episodes//2:]):.2f}\n"
    )

    # Plot episode vs time steps
    plot_episode_vs_timesteps(time_steps, episode_numbers)
