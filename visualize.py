import matplotlib.pyplot as plt
import numpy as np


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
            elif (row, col) == env.start_state:
                policy_grid[row, col] = "S"
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
    """Plot the episodes vs timestep to show improvement over time"""
    plt.figure(figsize=(10, 7))
    plt.scatter(time_steps, episode_numbers, s=3, color="r", label="red")
    plt.ylabel("Episodes")
    plt.xlabel("Time steps")
    plt.title("Windy Gridworld Learning")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_multiple_episode_vs_timesteps(alpha_epsilon_comb_dict):
    """Plot the episodes vs timestep for each combination of alpha, epsilon
    to show improvement over time and hyperparameter comparison"""
    plt.figure(figsize=(10, 7))
    colors = ["b", "r", "g", "y", "m", "c"]
    i = 0

    for key, value in alpha_epsilon_comb_dict.items():
        alpha, epsilon = key
        time_steps, episode_numbers = value
        param_label = f"α = {alpha}, ε = {epsilon}"
        plt.plot(
            time_steps, episode_numbers, color=colors[i], label=param_label, linewidth=3
        )
        i += 1

    plt.ylabel("Episodes")
    plt.xlabel("Time steps")
    plt.title("Windy Gridworld Learning")
    plt.grid(True)
    plt.legend()
    plt.show()
