import numpy as np
import matplotlib.pyplot as plt

# Hyperparams
ALPHA = 0.01
ITER = 10_000

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


class MultiplayerGames:
    """MultiplayerGames performs an every-visit update algorithm using policy iteration"""

    def __init__(
        self, p1_reward, isPrisoner, p1_start, p2_start, alpha=ALPHA, iterations=ITER
    ):
        self.alpha = alpha
        self.iterations = iterations

        self.p1_reward = np.array(p1_reward)

        if isPrisoner:
            # Prisoner's dilemma reward for player 2 is transpose
            self.p2_reward = self.p1_reward.T
        else:
            # For other games, player 2 reward is the negation
            self.p2_reward = -self.p1_reward

        self.num_actions_p1 = len(p1_reward)
        self.num_actions_p2 = len(p1_reward[0])

        self.p1_policy = np.array(p1_start)
        self.p2_policy = np.array(p2_start)

        # Store copies of the current
        self.p1_progress = [self.p1_policy.copy()]
        self.p2_progress = [self.p2_policy.copy()]

    def update(self, policy, action, reward):
        updated_policy = policy.copy()

        for a in range(len(policy)):
            # Update selected action separately from others
            if a == action:
                updated_policy[a] += self.alpha * reward * (1 - policy[a])
            else:
                updated_policy[a] -= self.alpha * reward * policy[a]

        # Keep the policy probabilities between 0 and 1, and normalize
        updated_policy = np.clip(updated_policy, 0, 1)
        updated_policy /= updated_policy.sum()
        return updated_policy

    def play(self):
        for _ in range(self.iterations):
            # Choose action according to prob. dist.
            a1 = np.random.choice(self.num_actions_p1, p=self.p1_policy)
            a2 = np.random.choice(self.num_actions_p2, p=self.p2_policy)

            p1_reward = self.p1_reward[a1, a2]
            p2_reward = self.p2_reward[a1, a2]

            self.p1_policy = self.update(self.p1_policy, a1, p1_reward)
            self.p2_policy = self.update(self.p2_policy, a2, p2_reward)

            # Save progress for plotting
            self.p1_progress.append(self.p1_policy.copy())
            self.p2_progress.append(self.p2_policy.copy())

        # Get total game value from both players' perspective
        p1_value = 0
        p2_value = 0
        for i in range(self.num_actions_p1):
            for j in range(self.num_actions_p2):
                p1_value += self.p1_policy[i] * self.p2_policy[j] * self.p1_reward[i, j]
                p2_value += self.p1_policy[i] * self.p2_policy[j] * self.p2_reward[i, j]

        return self.p1_policy, self.p2_policy, p1_value, p2_value

    def plot_policies(self, title):
        p1_progress = np.array(self.p1_progress)
        p2_progress = np.array(self.p2_progress)

        plt.figure(figsize=(12, 8))

        # Plotting player 1 policy
        plt.subplot(2, 1, 1)
        for i in range(self.num_actions_p1):
            plt.plot(p1_progress[:, i], label=f"Action {i}")
        plt.title(f"Player 1 Policy - {title}")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)

        # Plotting player 2 policy
        plt.subplot(2, 1, 2)
        for i in range(self.num_actions_p2):
            plt.plot(p2_progress[:, i], label=f"Action {i}")
        plt.title(f"Player 2 Policy - {title}")
        plt.xlabel("Iterations")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    for game_name, reward_matrix in games.items():
        print("\n", game_name)

        for i, (p1_start, p2_start) in enumerate(start_policy[game_name]):

            game = MultiplayerGames(
                reward_matrix,
                True if game_name is PRISONER else False,
                p1_start,
                p2_start,
            )
            p1_final, p2_final, p1_value, p2_value = game.play()

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
