import os, sys
SRC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir)
sys.path.append(SRC_DIR)

from src.envs import GridWorld

import numpy as np

class PolicyEvaluation:

    def __init__(self, policy=None):
        
        # Initialize GridWorld environment.
        self.environment= GridWorld(num_rows=4, num_columns=4)

        # Initialize policy.
        if policy is None:
            self.policy = np.ones([self.environment.num_states, self.environment.num_actions]) / self.environment.num_actions
        print (self.environment.states)

        # Initialize variables for policy evaluation.
        self.v = np.zeros(self.environment.num_states)

    def run(self, discount=1.0, theta=0.00001):

        # Run policy evaluation.
        while True:
            threshold = 0
            v = self.v.copy()
            for state in self.environment.state_space:
                value_temp = 0
                for action_idx, action in enumerate(self.environment.action_space):
                    reward, next_state = self.environment.step(action=action, state=state)
                    value_temp += self.policy[state, action_idx] * (reward + (discount * v[next_state]))
                self.v[state] = value_temp
                threshold = max(threshold,np.max(np.abs(v - self.v[state])))
            print self.v, threshold
            if threshold < theta:
                break
            

if __name__=="__main__":
    pe = PolicyEvaluation()
    pe.run()