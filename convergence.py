import windygridworld
import visualize
import click
import control_algorithms


OPTIMAL_PATH_LENGTH = 15

if __name__ == "__main__":
    env = windygridworld.WindyGridworld()
    runs = 50
    total_time_steps = 0
    total_episodes = 0

    print("\nSelect one of Sarsa[s], Q-Learning[q], Q(λ) [ql], and Sarsa(λ) [sl]:")
    options = {"s": "Sarsa", "q": "Q-Learning", "sl": "Sarsa(λ)", "ql": "Q(λ)"}
    control_type = click.prompt("Choose one", type=click.Choice(options.keys()))

    for i in range(1, runs + 1):
        if control_type == "s":
            Q, _, time_steps, episode_numbers = control_algorithms.sarsa_to_convergence(
                env, OPTIMAL_PATH_LENGTH
            )
        elif control_type == "q":
            Q, _, time_steps, episode_numbers = (
                control_algorithms.q_learning_to_convergence(env, OPTIMAL_PATH_LENGTH)
            )
        elif control_type == "sl":
            # TODO
            pass
        elif control_type == "ql":
            # TODO
            pass

        curr_total_steps = time_steps[-1]
        total_time_steps += curr_total_steps

        curr_total_episodes = episode_numbers[-1]
        total_episodes += curr_total_episodes

        print(
            f"Run {i}/{runs}. Convergence reached in:\n{curr_total_steps} time steps\n{curr_total_episodes} episodes\n"
        )

    print(
        f"{options[control_type]} average convergence metrics over {runs} runs: \
        \nAverage time steps: {total_time_steps/runs} \
        \nAverage episodes: {total_episodes/runs}\n"
    )

    # Visualize the learned policy
    print("Policy")
    visualize.visualize_policy(env, Q)

    # Plot episode vs time steps
    visualize.plot_episode_vs_timesteps(time_steps, episode_numbers)
