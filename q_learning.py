import numpy as np
import windygridworld
import visualize


def q_learning(env, episodes, alpha, gamma, epsilon):
    """Q-learning off-policy control"""
    # Q holds all values for (row, col, action) triplets
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

        # Continue until terminal (goal) state reached
        while not done:
            # Choose action using epsilon-greedy selection
            action = env.epsilon_greedy_policy(Q, state, epsilon, env.num_actions)

            # Take an action, observe reward and new state
            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1
            total_steps += 1

            time_steps.append(total_steps)
            episode_numbers.append(episode)

            # Q-learning update rule
            greedy_path_size = np.argmax(Q[next_state[0], next_state[1], :])
            Q[state[0], state[1], action] += alpha * (
                reward
                + gamma * Q[next_state[0], next_state[1], greedy_path_size]
                - Q[state[0], state[1], action]
            )

            # Move to next state
            state = next_state

        steps[episode] = step_count

    return Q, steps, time_steps, episode_numbers


if __name__ == "__main__":
    # Create environment
    num_episodes = 170

    alpha_epsilon_comb_dict = dict.fromkeys(
        [
            (0.5, 0.1),
            (1e-10, 0.1),
            (0.5, 1e-10),
            (1e-10, 1e-10),
            (1 - 1e-10, 1 - 1e-10),
        ],
        None,
    )

    for alpha, epsilon in alpha_epsilon_comb_dict.keys():
        # Train for different combinations of epsilon and alpha
        env = windygridworld.WindyGridworld()

        print(f"Alpha = {alpha}\nEpsilon = {epsilon}\n")

        # Train agent with Q-learning
        Q, steps, time_steps, episode_numbers = q_learning(
            env, episodes=num_episodes, alpha=0.5, gamma=1.0, epsilon=0.1
        )
        # Visualize the learned policy
        print("Q-Learning")
        visualize.visualize_policy(env, Q)

        optimal_path_size = env.get_optimal_path_size(Q)
        print("Optimal path length:\n", optimal_path_size)

        average_of_episodes = num_episodes // 2
        print(
            f"Average steps over last {average_of_episodes} episodes:\n {np.mean(steps[-num_episodes//2:]):.2f}\n"
        )

        alpha_epsilon_comb_dict[(alpha, epsilon)] = (time_steps, episode_numbers)

    # Plot episode vs time steps for each alpha, epsilon combo
    visualize.plot_multiple_episode_vs_timesteps(alpha_epsilon_comb_dict)
