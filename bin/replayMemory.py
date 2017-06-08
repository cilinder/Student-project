import numpy as np

class ReplayMemory:


    def __init__(self, params):

        self.max_size = 5000

        self.recent_mem_size = 1000

        self.state_dim = params.input_data_size
        self.n_actions = params.num_actions

        self.entries = 0


    def save_transition(self, action, currentState, nextState, reward):

        pass


