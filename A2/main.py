import numpy as np
import windygridworld
import visualize
import click
import control_algorithms


if __name__ == "__main__":
    # Define numnber of episodes and combinations of hyperparams alpha, epsilon
    num_episodes = 170

    alpha_epsilon_comb_dict = dict.fromkeys(
        [
            (0.5, 0.1),
            (0.1, 0.1),
            (0.5, 1e-5),
            (0.1, 0.1),
            (0.5, 0.5),
        ],
        None,
    )

    print("\nSelect one of Sarsa[s], Q-Learning[q], Q(λ) [ql], and Sarsa(λ) [sl]:")
    options = {"s": "Sarsa", "q": "Q-Learning", "sl": "Sarsa(λ)", "ql": "Q(λ)"}
    control_type = click.prompt("Choose one", type=click.Choice(options.keys()))

    for env in [
        windygridworld.WindyGridworld(),
        windygridworld.StochasticGridWorld(),
    ]:
        # Test on both regular and stochastic gridworld environments
        env_type = env.get_name()

        for alpha, epsilon in alpha_epsilon_comb_dict.keys():
            # Train for different combinations of epsilon and alpha

            print(f"==========={env_type} Environment===========")

            print(f"Alpha = {alpha}\nEpsilon = {epsilon}\n")

            if control_type == "s":
                Q, steps, time_steps, episode_numbers = control_algorithms.sarsa(
                    env, episodes=num_episodes, alpha=alpha, epsilon=epsilon
                )
            elif control_type == "q":
                Q, steps, time_steps, episode_numbers = control_algorithms.q_learning(
                    env, episodes=num_episodes, alpha=alpha, epsilon=epsilon
                )
            elif control_type == "sl":
                Q, steps, time_steps, episode_numbers = control_algorithms.sarsa_lambda(
                    env, episodes=num_episodes, alpha=alpha, epsilon=epsilon
                )
            elif control_type == "ql":
                Q, steps, time_steps, episode_numbers = (
                    control_algorithms.watkins_q_lambda(
                        env, episodes=num_episodes, alpha=alpha, epsilon=epsilon
                    )
                )

            # Visualize the learned policy
            print(f"{options[control_type]} Policy")
            visualize.visualize_policy(env, Q)

            optimal_path_size = env.get_optimal_path_size(Q)
            if optimal_path_size == -1:
                print("Optimal path length > 1000")
            else:
                print("Optimal path length:\n", optimal_path_size)

            average_of_episodes = num_episodes // 2
            print(
                f"Average steps over last {average_of_episodes} episodes:\n {np.mean(steps[-num_episodes//2:]):.2f}\n"
            )

            alpha_epsilon_comb_dict[(alpha, epsilon)] = (time_steps, episode_numbers)

        # Plot episode vs time steps for each alpha, epsilon combo
        visualize.plot_multiple_episode_vs_timesteps(
            alpha_epsilon_comb_dict, env_type, options[control_type]
        )
