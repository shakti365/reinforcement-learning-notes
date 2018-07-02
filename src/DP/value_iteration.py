import os, sys
SRC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir)
sys.path.append(SRC_DIR)

from src.envs import GridWorld

import numpy as np

class ValueIteration:

    def __init__(self, policy=None):
        
        # Initialize GridWorld environment.
        self.environment= GridWorld(num_rows=4, num_columns=4)

        # Initialize policy.
        if policy is None:
            self.policy = np.ones([self.environment.num_states, self.environment.num_actions]) / self.environment.num_actions
        print (self.environment.states)

        # Initialize variables for policy evaluation.
        self.v = np.zeros(self.environment.num_states)

    
    def value_iteration(self, discount):

        # Iterate over all the states in state space.            
        for state in self.environment.state_space:

            # Initialize an empty list to store action value fucntion.
            action_value = []

            # Iterate over all possible actions from current state.
            for action_idx, action in enumerate(self.environment.action_space):
                
                # Perform action in the environment and observe reward and next state.
                reward, next_state = self.environment.step(action=action, state=state)

                # Calculate action value function for each action from current state.
                action_value.append((reward + (discount * self.v[next_state])))

            # Select best action from updated policy.
            best_action = np.argmax(action_value)

            # Update the value function with the current best value function from given state and action.
            self.v[state] = action_value[best_action]


    def run(self, discount=1.0, theta=0.00001):

        # Run policy evaluation.
        while True:
        # for _ in range(2):

            # Initialize stopping threshold.
            threshold = 0

            # Create copy of old value functions for all states.
            v = self.v.copy()

            # Call value iteration.
            self.value_iteration(discount)

            # Calculate max change in value functions from last state
            threshold = max(threshold, np.max(np.abs(v - self.v)))
            
            print self.v, threshold
            
            # Check if change in value function from last state to current is above the tolerance value and stop iteration.
            if threshold < theta:
                break

if __name__=="__main__":
    vi = ValueIteration()
    vi.run()