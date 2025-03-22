import click
import multi_game as mg

# Game types
PRISONER = "Prisoner's dilemma"
PENNIES = "Matching pennies"
RPS = "Rock paper scissors"

# Define the starting policies
policy_2_by_2 = [
    ([0.5, 0.5], [0.5, 0.5]),  # Uniform distribution
    ([0.8, 0.2], [0.8, 0.2]),  # Biased for both players to co-operate
    ([0.2, 0.8], [0.2, 0.8]),  # Biased for both players to defect
    ([0.8, 0.2], [0.2, 0.8]),  # Biased for one to co-op, one to defect
]
start_policy = {
    PRISONER: policy_2_by_2,
    PENNIES: policy_2_by_2,
    RPS: [
        ([1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]),  # Uniform
        ([0.8, 0.1, 0.1], [0.1, 0.8, 0.1]),  # Biased
        ([0.8, 0.1, 0.1], [0.8, 0.1, 0.1]),  # Biased
        ([0.1, 0.1, 0.8], [0.1, 0.8, 0.1]),  # Biased
    ],
}

# Game reward functions
games = {
    PRISONER: [[5, 0], [10, 1]],
    PENNIES: [[1, -1], [-1, 1]],
    RPS: [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
}


def main():

    print("\nSelect Basic Update[b] or Expected Update[e]:")
    update_options = {
        "b": mg.MultiplayerGames.basic_update,
        "e": mg.MultiplayerGames.expected_update,
    }
    update_f = update_options[
        click.prompt("Choose one", type=click.Choice(update_options.keys()))
    ]

    for game_name, reward_matrix in games.items():
        print(f"\n{game_name}")

        for i, (p1_start, p2_start) in enumerate(start_policy[game_name]):

            game = mg.MultiplayerGames(
                reward_matrix,
                True if game_name is PRISONER else False,
                p1_start,
                p2_start,
            )
            p1_final, p2_final, p1_value, p2_value = game.play(update_f)

            # Printing final results
            print(f"\nStart policy {i+1}:")

            print(f"P1 Initial: {p1_start}")
            print(f"P1 Final: {p1_final.round(4)}")

            print(f"P2 Initial: {p2_start}")
            print(f"P2 Final: {p2_final.round(4)}")

            print(f"P1 Value: {p1_value:.2f}")
            print(f"P2 Value: {p2_value:.2f}")

            game.plot_policies(f"{game_name} - Initial Distribution {i+1}")


if __name__ == "__main__":
    main()
