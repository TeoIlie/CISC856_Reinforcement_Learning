import numpy as np
import windygridworld
import visualize


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
        action = env.epsilon_greedy_policy(Q, state, epsilon, env.num_actions)

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
            next_action = env.epsilon_greedy_policy(
                Q, next_state, epsilon, env.num_actions
            )

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


if __name__ == "__main__":
    # Create environment
    env = windygridworld.WindyGridworld()
    num_episodes = 170

    # Train agent with Sarsa
    Q, steps, time_steps, episode_numbers = sarsa(
        env, episodes=num_episodes, alpha=0.5, gamma=1.0, epsilon=0.1
    )

    # Visualize the learned policy
    visualize.visualize_policy(env, Q)

    optimal_path_size = env.get_optimal_path_size(Q)
    print("Optimal path length:\n", optimal_path_size)

    average_of_episodes = num_episodes // 2
    print(
        f"Average steps over last {average_of_episodes} episodes:\n {np.mean(steps[-num_episodes//2:]):.2f}\n"
    )

    # Plot episode vs time steps
    visualize.plot_episode_vs_timesteps(time_steps, episode_numbers)
