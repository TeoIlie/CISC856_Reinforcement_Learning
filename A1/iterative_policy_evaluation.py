import numpy as np

"""Global parameters of the gridworld game"""
rows, cols = 5, 5 # Grid dimensions. Grid encoded as (row, col) tuples, with (0,0) in the top-left

def get_next_state_reward(state, action):
    """Given a state and action, return new state and reward"""
    row, col = state
    row_inc, col_inc = action

    # Calculate next state position
    next_state_row, next_state_col = row + row_inc, col + col_inc
    
    # A and B are special states that teleport agent to A', B' and yield 
    # a special reward
    if state == (0,1): # state A
        return (4,1), 10
    elif state == (0,3): # state B
        return (2,3), 5
    
    # Return reward 0 for a valid move, and -1 for hitting a wall
    if 0 <= next_state_row < rows and 0 <= next_state_col < cols:
        return (next_state_row, next_state_col), 0  # Valid move with reward 0
    else:
        return state, -1  # Hitting a wall

def iterative_policy_eval(gamma):
    """Iterative policy evaluation strategy of Dynamic Programming, to compute state-value function v, with discount rate gamma.
    Based on pseudocode from p.83 of Sutton & Barto. Retuns array V with values, and total state space sweeps."""

    V = np.random.uniform(-20, 20, (rows, cols)) # initialize all state values in V arbitrarily in range [-20, 20]
    actions = [(-1,0), (0,1), (1,0), (0,-1)] # action options: North, East, South, West
    theta = 1e-6  # theta, the convergence precision

    delta = theta + 1
    total_sweeps = 0

    while delta >= theta:
        
        delta = 0
        new_V = np.copy(V)
        
        for i in range(rows):
            for j in range(cols):
                # loop for every state in V
                state = (i, j)
                action_values = []
                
                for action in actions:
                    # take all 4 actions, calculating their value according to the update rule
                    # that comes from the Bellman optimality equation
                    next_state, reward = get_next_state_reward(state, action)
                    action_values.append(reward + gamma * V[next_state])
                
                # take the weighted average of the options, with probability 1/4 to move in all 4 directions
                # taking the maximum value instead here would make the algorithm use Value Iteration
                new_V[state] = 0.25 * action_values[0] + 0.25 * action_values[1] \
                             + 0.25 * action_values[2] + 0.25 * action_values[3]

                delta = max(delta, abs(new_V[state] - V[state]))

        total_sweeps += 1
        V = new_V

    return V, total_sweeps

if __name__ == "__main__":
# Calculate the state-value array for each value of gamma discount rate
    gamma_values = [0.75, 0.85, 0.9]
    for gamma in gamma_values:  
        V, total_iter = iterative_policy_eval(gamma)
        print(f"\nState values for discount rate = {gamma}:\nTotal iterations = {total_iter}\n", V)
