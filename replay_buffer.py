import numpy as np
import joblib
import d4rl

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, ac_dim, goal_dim, max_steps, get_goal_from_state, compute_reward):

        self.entries = int(buffer_size / max_steps)
        self.obs = np.zeros((self.entries, max_steps, state_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.entries, max_steps, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.entries, max_steps, ac_dim), dtype=np.float32)
        self.goals = np.zeros((self.entries, max_steps, goal_dim), dtype=np.float32)
        self.rewards = np.zeros((self.entries, max_steps, 1), dtype=np.float32)
        self.traj_len = np.zeros((self.entries), dtype=np.int32)

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.ac_dim = ac_dim
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self.get_goal_from_state = get_goal_from_state
        self.compute_reward = compute_reward

        self.is_full = False
        self.curr_ptr = 0

    def push(self, input):
        traj_len = input['observations'].shape[0]
        self.obs[self.curr_ptr, :traj_len] = input['observations']
        self.next_obs[self.curr_ptr, :traj_len] = input['next_observations']
        self.actions[self.curr_ptr, :traj_len] = input['actions']
        self.goals[self.curr_ptr, :traj_len] = input['desired_goals']
        self.rewards[self.curr_ptr, :traj_len] = input['rewards']
        self.traj_len[self.curr_ptr] = traj_len

        self.curr_ptr += 1
        if self.curr_ptr == self.entries:
            self.curr_ptr = 0
            self.is_full = True
      
    def load_d4rl(self, env):
        if self.goal_dim == 0:
            dataset = d4rl.qlearning_dataset(env)
            size = dataset['observations'].shape[0]
            start = 0
            for i in range(0, size-self.max_steps, self.max_steps):
                self.push({'observations': dataset['observations'][i:i+self.max_steps],
                           'next_observations': dataset['observations'][i+1:i+1+self.max_steps],
                           'actions': dataset['actions'][i:i+self.max_steps],
                           'desired_goals': np.zeros((self.max_steps, 0)),
                           'rewards': dataset['rewards'][i:i+self.max_steps][..., np.newaxis]
                            })
            
        else:
            dataset = env.get_dataset()
            size = dataset['observations'].shape[0]
            start = 0
            for i in range(0, size):
                if dataset['timeouts'][i]:
                    if i - start > 1:
                        self.push({'observations': dataset['observations'][start:i],
                                   'next_observations': dataset['observations'][start+1:i+1],
                                   'actions': dataset['actions'][start:i],
                                   'desired_goals': dataset['infos/goal'][start:i],
                                   'rewards': dataset['rewards'][start:i, np.newaxis]
                                    })
                    start = i + 1  
                
    def load_dataset(self, path):
        if '.pkl' not in path:
            path = path + '.pkl'
        with open(path, 'rb') as f:
            loaded_data = joblib.load(f)
        size = loaded_data['o'].shape[0]
        for i in range(size):
            self.push({'observations': loaded_data['o'][i, :-1],
                       'next_observations': loaded_data['o'][i, 1:],
                       'actions': loaded_data['u'][i],
                       'desired_goals': loaded_data['g'][i],
                       })

    def get_bounds(self):
        boarder = self.entries if self.is_full else self.curr_ptr
        return 0, boarder

    def sample(self, batch_size, her_prob=0):
        _, high = self.get_bounds()
        entry = np.random.randint(0, high, batch_size)
        index = np.random.randint(0, self.traj_len[entry])
        ret_obs = self.obs[entry, index]
        ret_next_obs = self.next_obs[entry, index]
        ret_actions = self.actions[entry, index]
        ret_rewards = self.rewards[entry, index]

        # hindsight experience replay
        goal_index = np.random.randint(index, self.traj_len[entry], batch_size)
        replace = (np.random.rand(batch_size) < her_prob).astype(np.int32)
        ret_goals = self.get_goal_from_state(self.next_obs[entry, goal_index]) * replace[..., np.newaxis] + \
                    self.goals[entry, 0] * (1 - replace[..., np.newaxis])
        
        info = {}
        info['rewards'] = ret_rewards
        return ret_obs, ret_actions, ret_next_obs, ret_goals, info

    def sample_sequence(self, batch_size, tau):
        _, high = self.get_bounds()
        entry = np.random.randint(0, high, batch_size)
        index = np.random.randint(0, self.traj_len[entry] - tau, batch_size)
        ret_obs = np.concatenate([self.obs[entry, np.newaxis, index + t] for t in range(tau)], axis=1)
        ret_next_obs = np.concatenate([self.next_obs[entry, np.newaxis, index + t] for t in range(tau)], axis=1)
        ret_actions = np.concatenate([self.actions[entry, np.newaxis, index + t] for t in range(tau)], axis=1)
        ret_rewards = np.concatenate([self.rewards[entry, np.newaxis, index + t] for t in range(tau)], axis=1)
        goal_index = np.random.randint(index + tau, self.traj_len[entry], batch_size)
        ret_goals = self.get_goal_from_state(self.obs[entry, np.newaxis, goal_index]).repeat(tau, axis=1)
        info = {}
        info['rewards'] = ret_rewards
        return ret_obs, ret_actions, ret_next_obs, ret_goals, info


    def sample_inter_reanalysis(self):
        while True:
            _, high = self.get_bounds()
            state_traj = np.random.randint(0, high)
            state_idx = np.random.randint(0, self.traj_len[state_traj])

            goal_traj = np.random.randint(0, high)
            goal_idx = np.random.randint(0, self.traj_len[goal_traj])

            if self.compute_reward(self.get_goal_from_state(self.obs[state_traj, state_idx]),
                                   self.goals[goal_traj, goal_idx], None) == 0:
                break
        return self.obs[state_traj, state_idx], self.goals[goal_traj, goal_idx]

    def sample_intra_reanalysis(self):
        while True:
            _, high = self.get_bounds()
            entry = np.random.randint(0, high)
            start = np.random.randint(0, self.traj_len[entry]-1)
            end = np.random.randint(start+1, self.traj_len[entry])
            if self.compute_reward(self.get_goal_from_state(self.obs[entry, start]),
                                   self.get_goal_from_state(self.next_obs[entry, end]), None) == 0:
                break
        return self.obs[entry][start:end], self.actions[entry][start:end], self.next_obs[entry][start:end], \
               self.get_goal_from_state(self.next_obs[entry, end, np.newaxis]).repeat(end-start, axis=0)