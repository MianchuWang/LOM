import torch
import numpy as np

from agents.td3bc import TD3BC 
from networks.networks import Policy, Qnetwork

class STR(TD3BC):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.behaviour_policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.behaviour_policy_opt = torch.optim.Adam(self.behaviour_policy.parameters(), lr=3e-4)
        for _ in range(20):
            self.train_behaviour_policy(batch_size=256)
        self.policy.load_state_dict(self.behaviour_policy.state_dict())
            
    def train_behaviour_policy(self, batch_size):
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
            q_values = self.q_nets[0](states_prep, actions_prep)
            curr_q_values = self.q_nets[0](states_prep, gen_actions)
            advs = q_values - curr_q_values
            weights = torch.clip(torch.exp(2 * advs), -10000, 100)
        policy_loss = ((gen_actions - actions_prep).pow(2).mean(dim=1) * weights.squeeze()).mean()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        return {'policy_loss': policy_loss.item()}