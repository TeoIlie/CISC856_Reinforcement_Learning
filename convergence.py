import numpy as np
import windygridworld
import visualize
import click
import control_algorithms


OPTIMAL_PATH_LENGTH = 15

if __name__ == "__main__":
    env = windygridworld.WindyGridworld()
    runs = 20
    total_time_steps = 0
    total_episodes = 0

    for i in range(1, runs + 1):
        Q, _, time_steps, episode_numbers = control_algorithms.sarsa_to_convergence(
            env, OPTIMAL_PATH_LENGTH
        )

        curr_total_steps = time_steps[-1]
        total_time_steps += curr_total_steps

        curr_total_episodes = episode_numbers[-1]
        total_episodes += curr_total_episodes

        print(
            f"Run {i}/{runs}. Convergence reached in:\n{curr_total_steps} time steps\n{curr_total_episodes} episodes\n"
        )

    print(
        f"Average convergence metrics over {runs} runs: \
        \nAverage time steps: {total_time_steps/runs} \
        \nAverage episodes: {total_episodes/runs}\n"
    )

    # Visualize the learned policy
    print(f"Sarsa Optimal Policy")
    visualize.visualize_policy(env, Q)

    # Plot episode vs time steps
    visualize.plot_episode_vs_timesteps(time_steps, episode_numbers)
