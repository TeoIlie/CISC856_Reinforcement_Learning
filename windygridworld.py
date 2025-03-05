import random
import numpy as np

# Constants define environment type
REGULAR = "Regular"  # Normal windy gridworld from Sutton & Barto
STOCHASTIC = "Stochastic"  # King's moves, stochastic wind


class WindyGridworld:
    """WindyGridworld defines an environment for an agent according
    to Sutton & Barto pg. 130"""

    def __init__(self):
        # Grid size 7 x 10
        self.rows = 7
        self.cols = 10

        # Reward always -1
        self.reward = -1

        # Start, goal positions
        self.start_state = (3, 0)
        self.goal_state = (3, 7)

        # Current position
        self.state = self.start_state

        # Upward wind strength for each column
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        # Actions: up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.num_actions = len(self.actions)

    def get_name(self):
        return REGULAR

    def reset(self):
        # Reset to start state
        self.state = self.start_state
        return self.start_state

    def wind_movement(self, col):
        return self.wind[col]

    def step(self, action):
        """step function takes an action and applies it to the agent,
        given wind and that the agent cannot go outside the gridworld."""

        row, col = self.state

        row_move, col_move = self.actions[action]

        next_row = row + row_move
        next_col = col + col_move

        # Account for wind effect
        next_row = next_row - self.wind_movement(col)

        # Keep values in gridworld bounds
        next_row = max(0, min(next_row, self.rows - 1))
        next_col = max(0, min(next_col, self.cols - 1))

        next_state = (next_row, next_col)

        # Check if goal state reached
        if next_state == self.goal_state:
            done = True
        else:
            done = False

        self.state = next_state
        return next_state, self.reward, done

    def get_optimal_path_size(self, Q):
        """Get the length of the optimal path with respect to the current policy π
        that is greedy with respect to Q"""
        # Start from the start state
        current_state = self.reset()
        path_length = 0
        max_path_size = 1000

        while path_length < max_path_size:
            best_action = np.argmax(Q[current_state[0], current_state[1], :])

            current_state, r, done = self.step(best_action)
            path_length += 1

            if done:
                break

        if path_length == max_path_size:
            return f"Optimal path length greater than {max_path_size}"
        else:
            return str(path_length)

    def epsilon_greedy_policy(self, Q, state, epsilon, num_actions):
        """Choose non-greedy action randomly with epislon prob, and
        greedy action the rest of the time"""
        if random.uniform(0, 1) < epsilon:
            # Random action
            return random.randint(0, num_actions - 1)
        else:
            # Greedy action
            return np.argmax(Q[state[0], state[1]])


class StochasticGridWorld(WindyGridworld):
    """StochasticGridworld defines an environment for an agent according
    to Sutton & Barto pg. 130, except the agent can move diagonally as well
    (King's moves) and the wind is stochastic. This class inherits WindyGridworld
    and modifies these elements."""

    def __init__(self, *args, **kwargs):
        # Everything is the same except action options and wind
        super().__init__(*args, **kwargs)

        # Actions:  up, down, left, right
        # diagonally: up-left, up-right, down-left, down-right
        self.actions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        self.num_actions = len(self.actions)

    def get_name(self):
        return STOCHASTIC

    def wind_movement(self, col):
        """Add stochastic wind"""
        mean = self.wind[col]
        mean += random.choice([1, 0, -1])
        return mean
