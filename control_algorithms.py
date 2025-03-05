import numpy as np

# Default values for hyperparameters
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1.0


def sarsa(env, episodes, alpha, gamma, epsilon):
    """Use Sarsa control algorithm for given number of episodes."""
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


def sarsa_to_convergence(
    env, optimal_path_length, alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA
):
    """Sarsa control algorithm until optimal path reached."""
    # Q hold all values for (row, col, action) triplets
    Q = np.zeros((env.rows, env.cols, env.num_actions))

    # Keep track of stats for graphing
    time_steps = []
    episode_numbers = []
    total_steps = 0
    steps = []
    episode = 0

    # Loop to convergence
    while env.get_optimal_path_size(Q) != optimal_path_length:
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

        steps.append(step_count)
        episode += 1

    return Q, steps, time_steps, episode_numbers


def q_learning(env, episodes, alpha, gamma, epsilon):
    """Q-learning off-policy control for given number of episodes"""
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
            greedy_next_action = np.argmax(Q[next_state[0], next_state[1], :])
            Q[state[0], state[1], action] += alpha * (
                reward
                + gamma * Q[next_state[0], next_state[1], greedy_next_action]
                - Q[state[0], state[1], action]
            )

            # Move to next state
            state = next_state

        steps[episode] = step_count

    return Q, steps, time_steps, episode_numbers


def q_learning_to_convergence(
    env, optimal_path_length, alpha=ALPHA, epsilon=EPSILON, gamma=GAMMA
):
    """Q-learning algorithm until convergence reached"""
    # Q holds all values for (row, col, action) triplets
    Q = np.zeros((env.rows, env.cols, env.num_actions))

    # Keep track of stats for graphing
    time_steps = []
    episode_numbers = []
    total_steps = 0
    steps = []
    episode = 0

    # Loop to convergence
    while env.get_optimal_path_size(Q) != optimal_path_length:
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
            greedy_next_action = np.argmax(Q[next_state[0], next_state[1], :])
            Q[state[0], state[1], action] += alpha * (
                reward
                + gamma * Q[next_state[0], next_state[1], greedy_next_action]
                - Q[state[0], state[1], action]
            )

            # Move to next state
            state = next_state

        steps.append(step_count)
        episode += 1

    return Q, steps, time_steps, episode_numbers
