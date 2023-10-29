import torch
import numpy as np
from torch.distributions.normal import Normal

from agents.td3bc import TD3BC 
from networks.networks import Policy, Qnetwork

class EXPLORATION(TD3BC):
    def __init__(self, K=4, bc_initialization=50000, **agent_params):
        super().__init__(K=K, **agent_params)
        self.behaviour_policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.behaviour_policy_opt = torch.optim.Adam(self.behaviour_policy.parameters(), lr=3e-4)
        self.train_behaviour_policy(batch_size=256, steps=bc_initialization)
        self.policy.load_state_dict(self.behaviour_policy.state_dict())
            
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
        
        # Exploitation
        gen_dist, gen_actions = self.policy(states_prep)
        #gen_dist = Normal(loc=gen_actions, scale=0.1)
        with torch.no_grad():
            values_1 = self.q_nets[0](states_prep, actions_prep)
            curr_values_1 = self.q_nets[0](states_prep, gen_actions)
            advs_1 = values_1 - curr_values_1
            weights_exp_1 = torch.clip(torch.exp(2 * advs_1), None, 100).squeeze()
        policy_loss_1 = ((gen_actions - actions_prep).pow(2).mean(dim=1) * weights_exp_1).mean()
        
        # Exploration
        with torch.no_grad():
            sampled_actions = gen_dist.sample()
            sampled_actions = torch.clip(sampled_actions, -1, 1)
            
            values_2 = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for k in range(self.K):
                values_2[k] = self.q_nets[k](states_prep, sampled_actions)
            values_2 = torch.min(values_2, dim=0)[0]
            
            curr_values_2 = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for k in range(self.K):
                curr_values_2[k] = self.q_nets[k](states_prep, gen_actions)
            curr_values_2 = torch.min(curr_values_2, dim=0)[0]
            
            advs_2 = values_2 - curr_values_2
            weights_exp_2 = torch.clip(torch.exp(2 * advs_2), None, 100).squeeze()
        policy_loss_2 = ((gen_actions - sampled_actions).pow(2).mean(dim=1) * weights_exp_2).mean()
        
        policy_loss = policy_loss_1 + policy_loss_2
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        return {'policy/loss': policy_loss.item(),
                'policy/weights1': weights_exp_1.mean().item(),
                'policy/weights2': weights_exp_2.mean().item()}