import numpy as np
import matplotlib.pyplot as plt


class MultiplayerGames:
    def __init__(self, R1, alpha=0.01, iterations=10000):
        self.R1 = np.array(R1)
        self.num_actions_p1, self.num_actions_p2 = self.R1.shape

        # Set player 2's reward matrix based on game type
        if self.R1.shape == (2, 2) and np.array_equal(
            self.R1, np.array([[5, 0], [10, 1]])
        ):
            self.R2 = self.R1.T  # Prisoners Dilemma
        else:
            self.R2 = -self.R1  # Zero-sum games

        self.alpha = alpha
        self.iterations = iterations

    def initialize_policies(self, p1_init=None, p2_init=None):
        # Initialize with uniform or specified policies
        self.p1 = (
            np.ones(self.num_actions_p1) / self.num_actions_p1
            if p1_init is None
            else np.array(p1_init)
        )
        self.p2 = (
            np.ones(self.num_actions_p2) / self.num_actions_p2
            if p2_init is None
            else np.array(p2_init)
        )
        self.p1_history = [self.p1.copy()]
        self.p2_history = [self.p2.copy()]

    def update_policy(self, policy, action, reward):
        new_policy = policy.copy()
        # Apply the update rule from the assignment
        for a in range(len(policy)):
            if a == action:
                new_policy[a] += self.alpha * reward * (1 - policy[a])
            else:
                new_policy[a] -= self.alpha * reward * policy[a]

        # Normalize to ensure valid probability distribution
        return np.clip(new_policy, 0, 1) / np.sum(np.clip(new_policy, 0, 1))

    def run(self):
        rewards_p1, rewards_p2 = [], []

        for _ in range(self.iterations):
            # Select actions
            a1 = np.random.choice(self.num_actions_p1, p=self.p1)
            a2 = np.random.choice(self.num_actions_p2, p=self.p2)

            # Calculate rewards
            r1, r2 = self.R1[a1, a2], self.R2[a1, a2]

            # Update policies
            self.p1 = self.update_policy(self.p1, a1, r1)
            self.p2 = self.update_policy(self.p2, a2, r2)

            # Store history
            self.p1_history.append(self.p1.copy())
            self.p2_history.append(self.p2.copy())
            rewards_p1.append(r1)
            rewards_p2.append(r2)

        # Calculate game value
        value = sum(
            self.p1[i] * self.p2[j] * self.R1[i, j]
            for i in range(self.num_actions_p1)
            for j in range(self.num_actions_p2)
        )

        return self.p1, self.p2, value, rewards_p1, rewards_p2

    def plot_policies(self, title):
        p1_history = np.array(self.p1_history)
        p2_history = np.array(self.p2_history)

        plt.figure(figsize=(10, 6))

        # Player 1 policy plot
        plt.subplot(2, 1, 1)
        for i in range(self.num_actions_p1):
            plt.plot(p1_history[:, i], label=f"Action {i}")
        plt.title(f"Player 1 Policy - {title}")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)

        # Player 2 policy plot
        plt.subplot(2, 1, 2)
        for i in range(self.num_actions_p2):
            plt.plot(p2_history[:, i], label=f"Action {i}")
        plt.title(f"Player 2 Policy - {title}")
        plt.xlabel("Iterations")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Define the games
games = {
    "Prisoner's dilemma": [[5, 0], [10, 1]],
    "Matching pennies": [[1, -1], [-1, 1]],
    "Rock, paper, scissors": [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
}

# Initial policies for testing
init_policies = {
    "Prisoners Dilemma": [
        (None, None),
        ([0.8, 0.2], [0.8, 0.2]),
        ([0.2, 0.8], [0.2, 0.8]),
    ],
    "Matching Pennies": [(None, None), ([0.8, 0.2], [0.8, 0.2])],
    "Rock Paper Scissors": [(None, None), ([0.6, 0.2, 0.2], [0.2, 0.6, 0.2])],
}


def main():
    for game_name, reward_matrix in games.items():
        print(f"\n===== {game_name} =====")

        for i, (p1_init, p2_init) in enumerate(init_policies[game_name]):
            # Create and run game
            game = MultiplayerGames(reward_matrix, alpha=0.01, iterations=10000)
            game.initialize_policies(p1_init, p2_init)
            final_p1, final_p2, value, _, _ = game.run()

            # Print results
            print(f"\nInitial Policy {i+1}:")
            print(f"P1 Initial: {p1_init if p1_init is not None else 'Uniform'}")
            print(f"P2 Initial: {p2_init if p2_init is not None else 'Uniform'}")
            print(f"P1 Final: {final_p1.round(4)}")
            print(f"P2 Final: {final_p2.round(4)}")
            print(f"Game Value: {value:.4f}")

            # Plot policy evolution
            game.plot_policies(f"{game_name} - Policy {i+1}")


if __name__ == "__main__":
    main()
