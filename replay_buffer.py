import numpy as np
import joblib
import d4rl
import warnings

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, ac_dim, discount):

        self.obs = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, ac_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.terminals = np.zeros((buffer_size, 1), dtype=np.float32)
        self.returns = np.zeros((buffer_size, 1), dtype=np.float32)
       
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.buffer_size = buffer_size

        self.is_full = False
        self.curr_ptr = 0
        self.discount = discount
        
        self.sequence_eligible_indices = None
      
    def load_dataset(self, dataset, compute_return=True):
        dataset_size = dataset['observations'].shape[0]
        if dataset_size > self.buffer_size:
            raise Exception('The dataset (size ' + str(dataset_size) + ') is ' + \
                            'larger than the buffer capacity (size ' + str(self.buffer_size) + ').')
        self.obs[:dataset_size] = dataset['observations']
        if 'next_observations' in dataset.keys(): 
            self.next_obs[:dataset_size] = dataset['next_observations']
        else:
            self.next_obs[:dataset_size-1] = dataset['observations'][1:]
        self.actions[:dataset_size] = dataset['actions']
        self.rewards[:dataset_size] = dataset['rewards'][..., np.newaxis]
        self.terminals[:dataset_size] = dataset['terminals'][..., np.newaxis]
        self.curr_ptr = dataset_size
        if dataset_size == self.buffer_size:
            self.is_full = True
            
        if compute_return:
            ret = 0
            for i in reversed(range(self.curr_ptr)):
                if self.terminals[i]:
                    ret = self.rewards[i]
                else:
                    ret = self.rewards[i] + ret * self.discount
                self.returns[i] = ret
            
   
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
    
    def sample_with_next_action(self, batch_size):
        boarder = self.get_bounds()
        index = np.random.randint(0, boarder, batch_size)
        ret_obs = self.obs[index]
        ret_next_obs = self.next_obs[index]
        ret_actions = self.actions[index]
        ret_next_actions = self.actions[index+1]
        ret_rewards = self.rewards[index]
        ret_terminals = self.terminals[index]
        return ret_obs, ret_actions, ret_rewards, ret_next_obs, ret_next_actions, ret_terminals
    
    def sample_with_returns(self, batch_size):
        boarder = self.get_bounds()
        index = np.random.randint(0, boarder, batch_size)
        ret_obs = self.obs[index]
        ret_next_obs = self.next_obs[index]
        ret_actions = self.actions[index]
        ret_rewards = self.rewards[index]
        ret_terminals = self.terminals[index]
        ret_returns = self.returns[index]

        return ret_obs, ret_actions, ret_rewards, ret_next_obs, ret_terminals, ret_returns
    
    def sample_sequences(self, batch_size, sequence_length):
        if self.sequence_eligible_indices is None:
            boarder = self.get_bounds()
            all_indices = np.arange(boarder)
            terminal_indices = np.where(self.terminals)[0]
            self.sequence_eligible_indices = all_indices
            for i in range(sequence_length):
                self.sequence_eligible_indices = np.setdiff1d(self.sequence_eligible_indices, terminal_indices-i)

        start_idx = np.random.choice(self.sequence_eligible_indices, batch_size)
        range_idx = start_idx[:, None] + np.arange(sequence_length)
        
        sequences_obs = self.obs[range_idx]
        sequences_actions = self.actions[range_idx]
    
        return sequences_obs, sequences_actions
 


