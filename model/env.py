import numpy as np


class Environment:
    def __init__(self, d_states, n_actions):
        self.d_states = d_states
        self.n_actions = n_actions

    def start(self):
        return np.zeros(self.d_states)

    def next_state_reward(self, a):
        return np.zeros(self.d_states), 0.0

    def ended(self):
        return True
