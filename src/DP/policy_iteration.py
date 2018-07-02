import os, sys
SRC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir)
sys.path.append(SRC_DIR)

from src.envs import GridWorld

import numpy as np

class PolicyIteration:

    def __init__(self, policy=None):
        
        # Initialize GridWorld environment.
        self.environment= GridWorld(num_rows=4, num_columns=4)

        # Initialize policy.
        if policy is None:
            self.policy = np.ones([self.environment.num_states, self.environment.num_actions]) / self.environment.num_actions
        print (self.environment.states)

        # Initialize variables for policy evaluation.
        self.v = np.zeros(self.environment.num_states)

    def policy_evaluation(self, discount):
        """Run polcy evaluation for one step."""
        # Iterate over all the states in state space.
        for state in self.environment.state_space:

            # Initialize temporary variable to calculate value function for current state
            value_temp = 0

            # Iterate over all possible actions from current state.
            for action_idx, action in enumerate(self.environment.action_space):

                # Perform action in the environment and observe reward and next state.
                reward, next_state = self.environment.step(action=action, state=state)

                # Calculate value function for the state given the current reward and policy
                value_temp += self.policy[state, action_idx] * (reward + (discount * self.v[next_state]))

            # Assign temporary value to value function of current state.
            self.v[state] = value_temp

    def policy_improvement(self, discount=1.0):
        """Improves the current policy by acting greedily over value functions"""

        # Initialize current policy as stable.
        policy_stable = True

        # Iterate over all the states in state space.
        for state in self.environment.state_space:

            # Save old action according to the policy for later comparison.
            old_action = np.argmax(self.policy[state])

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

            # Check if past best action is same as the current one, 
            # If it is not then consider current policy as unstable.
            if old_action != best_action:
                policy_stable = False
            
            # Update the policy.
            new_policy = np.zeros(self.environment.num_actions)
            new_policy[best_action] = 1.0
            self.policy[state] = new_policy
        
        return policy_stable

    def run(self, discount=1.0, theta=0.00001):

        # Run policy evaluation.
        while True:
        # for _ in range(2):

            # Initialize stopping threshold.
            threshold = 0

            # Create copy of old value functions for all states.
            v = self.v.copy()

            # Call policy evaluation.
            self.policy_evaluation(discount)

            # Call policy improvement.
            policy_stable = self.policy_improvement(discount)

            # If current policy is stable then stop execution else keep doing evaluation and improvement.
            if policy_stable:
                break

            # Calculate max change in value functions from last state
            threshold = max(threshold, np.max(np.abs(v - self.v)))
            
            print self.v, self.policy, threshold
            
            # Check if change in value function from last state to current is above the tolerance value and stop iteration.
            if threshold < theta:
                break

if __name__=="__main__":
    pi = PolicyIteration()
    pi.run()