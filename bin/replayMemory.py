import numpy as np

class ReplayMemory:

    def __init__(self, params):

        self.max_size = params.memory_max_size

        self.recent_mem_size = 1000

        self.state_dim = params.input_data_size
        self.num_actions = params.num_actions

        self.total_entries = 0
        self.next_index = 0

        self.current_states = np.zeros((self.max_size, self.state_dim))
        self.actions = np.zeros(self.max_size, dtype=np.int_)
        self.next_states = np.zeros((self.max_size, self.state_dim))
        self.rewards = np.zeros(self.max_size)

    def save_transition(self, current_state, action, reward, next_state):

        self.current_states[self.next_index] = current_state

        self.next_states[self.next_index] = next_state

        self.actions[self.next_index] = np.where(action == 1)[0]

        self.rewards[self.next_index] = reward

        self.total_entries += 1
        self.next_index += 1

        if self.next_index % self.max_size == 0:
            self.next_index = 0


    def sample_one(self):

        return self.sample(1)


    def sample(self, batch_size):

        if batch_size > self.total_entries:
            print('WARNING: batch size is greater than number of total elements in replay memory, returning only batch of size: ', self.total_entries)
            batch_size = self.total_entries

        batch_start = np.random.randint(np.minimum(self.max_size, self.total_entries) - batch_size + 1)
        batch_end = batch_start + batch_size 

        actions = np.zeros((batch_size, self.num_actions))
        actions[np.arange(batch_size), self.actions[batch_start : batch_end]] = 1
        current_states = self.current_states[batch_start : batch_end]
        next_states = self.next_states[batch_start : batch_end]
        rewards = self.rewards[batch_start : batch_end]

        return (current_states, actions, rewards, next_states, batch_size)


















