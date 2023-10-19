import torch
import numpy as np

from agents.base_agent import BaseAgent
from networks.networks import Policy, Qnetwork

class TD3BC(BaseAgent):
    def __init__(self, K=2, **agent_params):
        super().__init__(**agent_params)
        self.policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.target_policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        
        self.K = K
        self.q_nets, self.q_target_nets, self.q_net_opts = [], [], []
        for k in range(self.K):
            self.q_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets.append(Qnetwork(self.state_dim, self.ac_dim).to(device=self.device))
            self.q_target_nets[k].load_state_dict(self.q_nets[k].state_dict())
            self.q_net_opts.append(torch.optim.Adam(self.q_nets[k].parameters(), lr=3e-4))  
        
        self.policy_delay = 2
        self.policy_training_steps = 0
        self.training_steps = 0
        self.alpha = 2.5
    
    def get_action(self, state):
        with torch.no_grad():
            state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
            _, action = self.policy(state_prep)
        return action.cpu().numpy().squeeze()
    
    def train_models(self):
        policy_info = {}
        value_info = self.train_value_function(batch_size=256)
        if self.training_steps % self.policy_delay == 0:
            policy_info = self.train_policy(batch_size=256)
            self.update_target_nets(self.q_nets, self.q_target_nets)
            self.update_target_nets([self.policy], [self.target_policy])
        self.training_steps += 1
        return {**value_info, **policy_info}
    
    def update_target_nets(self, net, target_net):
        for k in range(len(net)):
            for param, target_param in zip(net[k].parameters(), target_net[k].parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
    
    def train_policy(self, batch_size):
        states, actions, rewards, next_states, terminals  = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        
        _, gen_actions = self.policy(states_prep)
        q_values = self.q_nets[0](states_prep, gen_actions)
        lam = self.alpha / q_values.abs().mean().detach()
        policy_loss = - lam * q_values.mean() + ((actions_prep - gen_actions) ** 2).mean()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        return {'policy/loss': policy_loss.item()}
    
    def train_value_function(self, batch_size):
        states, actions, rewards, next_states, terminals  = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        
        with torch.no_grad():
            _, target_actions = self.target_policy(next_states_prep)
            smooth_noise = torch.clamp(0.2 * torch.randn_like(target_actions), -0.5, 0.5)
            target_actions = torch.clamp(target_actions + smooth_noise, -1, 1)

            q_next_values = torch.zeros(self.K, batch_size, 1).to(device=self.device)
            for i in range(self.K):
                q_next_values[i] = self.q_target_nets[i](next_states_prep, target_actions)
            q_next_value = torch.min(q_next_values, dim=0)[0]
            target_q_value = rewards_prep + (1 - terminals_prep) * self.discount * q_next_value
        
        for k in range(self.K):
            pred_q_value = self.q_nets[k](states_prep, actions_prep)
            q_loss = ((target_q_value - pred_q_value)**2).mean()
            self.q_net_opts[k].zero_grad()
            q_loss.backward()
            self.q_net_opts[k].step()
        
        return {'Q/loss': q_loss.item(), 
                'Q/pred_value': pred_q_value.mean().item(), 
                'Q/target_value': target_q_value.mean().item()}
