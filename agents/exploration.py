import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from agents.td3bc import TD3BC 
from networks.networks import Policy, Qnetwork

class Advque:
    def __init__(self, size=50000):
        self.size = size 
        self.current_size = 0
        self.que = np.zeros(size)
        self.idx = 0
    
    def update(self, values):
        l = len(values)

        if self.idx + l <= self.size:
            idxes = np.arange(self.idx, self.idx+l)
        else:
            idx1 = np.arange(self.idx, self.size)
            idx2 = np.arange(0, self.idx+l -self.size)
            idxes = np.concatenate((idx1, idx2))
        self.que[idxes] = values.reshape(-1)

        self.idx = (self.idx + l) % self.size 
        self.current_size = min(self.current_size+l, self.size)

    def get(self, threshold):
        return np.percentile(self.que[:self.current_size], threshold)


class EXPLORATION(TD3BC):
    def __init__(self, K=4, bc_initialization=50000, **agent_params):
        super().__init__(K=K, **agent_params)
        self.behaviour_policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.behaviour_policy_opt = torch.optim.Adam(self.behaviour_policy.parameters(), lr=3e-4)
        self.train_behaviour_policy(batch_size=256, steps=bc_initialization)
        self.policy.load_state_dict(self.behaviour_policy.state_dict())
        
        self.adv_que = Advque()
        self.quantile_threshold = 0.0
        self.maximum_thre = 80
            
    def train_behaviour_policy(self, batch_size, steps):
        if steps > 0: print('Learning a Gaussian behaviour policy ... ')
        for _ in range(steps):
            states, actions, _, _, _ = self.replay_buffer.sample(batch_size=batch_size)
            states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
            ac_dist, ac_mean = self.behaviour_policy(states_prep)
            log_prob = ac_dist.log_prob(actions_prep)
            behaviour_policy_loss = - log_prob.mean()
            self.behaviour_policy_opt.zero_grad()
            behaviour_policy_loss.backward()
            self.behaviour_policy_opt.step()

    def train_policy(self, batch_size):
        states, actions, rewards, next_states, terminals  = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        
        gen_dist, gen_actions = self.policy(states_prep)
        with torch.no_grad():
            sampled_actions = gen_dist.sample()
            sampled_actions = torch.clip(sampled_actions, -1, 1)
            
            values_2 = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for k in range(self.K):
                values_2[k] = self.q_nets[k](states_prep, sampled_actions)
            value_uncertainty = values_2.std(dim=0)
            values_2 = torch.min(values_2, dim=0)[0]
            
            curr_values_2 = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for k in range(self.K):
                curr_values_2[k] = self.q_nets[k](states_prep, gen_actions)
            curr_values_2 = torch.min(curr_values_2, dim=0)[0]
            
            advs_2 = values_2 - curr_values_2
            weights_exp_2 = torch.clip(torch.exp(2 * advs_2), None, 100).squeeze()
            
            # Best-advantage weighting
            self.adv_que.update(advs_2.cpu().numpy())
            temp_threshold = self.adv_que.get(self.quantile_threshold)
            positives = torch.ones_like(advs_2)
            #positives[advs_2 < temp_threshold] = 0
            weights_exp_2 = weights_exp_2 * positives
            self.quantile_threshold = min(self.quantile_threshold + 0.0004, 
                                          self.maximum_thre)
            
        policy_loss = ((gen_actions - sampled_actions).pow(2).mean(dim=1) * weights_exp_2).sum() / positives.sum()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        # compute KL-divergence
        bc_dist, _ = self.behaviour_policy(states_prep)
        kl_div = kl_divergence(gen_dist, bc_dist)
        
        return {'policy/loss': policy_loss.item(),
                'policy/weights2': weights_exp_2.mean().item(),
                'policy/kl_divergence': kl_div.mean().item(),
                'policy/value_uncertainty': value_uncertainty.mean().item()}
