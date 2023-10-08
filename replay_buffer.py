import numpy as np
import joblib
import d4rl
import warnings

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, ac_dim):

        self.obs = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, ac_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.terminals = np.zeros((buffer_size, 1), dtype=np.float32)
       
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.buffer_size = buffer_size

        self.is_full = False
        self.curr_ptr = 0
      
    def load_dataset(self, dataset):
        dataset_size = dataset['observations'].shape[0]
        if dataset_size > self.buffer_size:
            raise Exception('The dataset (size ' + str(dataset_size) + ') is ' + \
                            'larger than the buffer capacity (size ' + str(self.buffer_size) + ').')
        self.obs[:dataset_size] = dataset['observations']
        self.next_obs[:dataset_size] = dataset['next_observations']
        self.actions[:dataset_size] = dataset['actions']
        self.rewards[:dataset_size] = dataset['rewards'][..., np.newaxis]
        self.terminals[:dataset_size] = dataset['terminals'][..., np.newaxis]
        self.curr_ptr = dataset_size
        if dataset_size == self.buffer_size:
            self.is_full = True
        
   
    def get_bounds(self):
        boarder = self.buffer_size if self.is_full else self.curr_ptr
        return boarder

    def sample(self, batch_size):
        boarder = self.get_bounds()
        index = np.random.randint(0, boarder, batch_size)
        ret_obs = self.obs[index]
        ret_next_obs = self.next_obs[index]
        ret_actions = self.actions[index]
        ret_rewards = self.rewards[index]
        ret_terminals = self.terminals[index]
        return ret_obs, ret_actions, ret_rewards, ret_next_obs, ret_terminals
