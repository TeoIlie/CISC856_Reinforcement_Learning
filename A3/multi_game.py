import numpy as np
import matplotlib.pyplot as plt

# Hyperparams
ALPHA = 0.001
ITER = 10_000


class MultiplayerGames:
    """MultiplayerGames performs an every-visit update algorithm using policy iteration"""

    def __init__(
        self,
        p1_reward,
        isPrisoner,
        p1_start,
        p2_start,
        update_f,
        alpha=ALPHA,
        iterations=ITER,
    ):
        self.alpha = alpha
        self.iterations = iterations
        self.update_f = update_f

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

        # Track expected values of the player policies
        self.p1_e = np.array(p1_start)
        self.p2_e = np.array(p2_start)

        # Store copies of the current
        self.p1_progress = [self.p1_policy.copy()]
        self.p2_progress = [self.p2_policy.copy()]

    def play(self):
        for _ in range(self.iterations):
            # Choose action according to prob. dist.
            a1 = np.random.choice(self.num_actions_p1, p=self.p1_policy)
            a2 = np.random.choice(self.num_actions_p2, p=self.p2_policy)

            p1_reward = self.p1_reward[a1, a2]
            p2_reward = self.p2_reward[a1, a2]

            # Update expected policies iteratively
            if MultiplayerGames.is_expected_update(self.update_f):
                self.p1_e = self.iterative_expected_update(self.p1_policy, self.p1_e)
                self.p2_e = self.iterative_expected_update(self.p2_policy, self.p2_e)

            self.p1_policy = self.update_f(
                self.alpha, self.p1_policy, self.p1_e, a1, p1_reward
            )
            self.p2_policy = self.update_f(
                self.alpha, self.p2_policy, self.p2_e, a2, p2_reward
            )

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

    def iterative_expected_update(self, policy, expected_policy):
        """Iteratively updates the expected values of the policy,
        given the current policy"""
        new_e = expected_policy.copy()
        for i in range(len(policy)):
            new_e[i] += self.alpha * (policy[i] - expected_policy[i])
        return new_e

    @staticmethod
    def basic_update(alpha, policy, _, action, reward):
        updated_policy = policy.copy()

        for i in range(len(policy)):
            # Update selected action separately from others
            if i == action:
                updated_policy[i] += alpha * reward * (1 - policy[i])
            else:
                updated_policy[i] -= alpha * reward * policy[i]

        # Keep the policy probabilities between 0 and 1, and normalize
        updated_policy = np.clip(updated_policy, 0, 1)
        updated_policy /= updated_policy.sum()
        return updated_policy

    @staticmethod
    def expected_update(alpha, policy, expected_policy, action, reward):
        updated_policy = policy.copy()

        for i in range(len(policy)):
            # Calculate additional expression with expected policy value
            expected_expression = alpha * (expected_policy[i] - policy[i])
            if i == action:
                # Update selected action separately from others
                updated_policy[i] += (
                    alpha * reward * (1 - policy[i]) + expected_expression
                )

            else:
                updated_policy[i] -= alpha * reward * policy[i] + +expected_expression

        # Keep the policy probabilities between 0 and 1, and normalize
        updated_policy = np.clip(updated_policy, 0, 1)
        updated_policy /= updated_policy.sum()
        return updated_policy

    @staticmethod
    def is_expected_update(update_f):
        return update_f.__name__ == "expected_update"
