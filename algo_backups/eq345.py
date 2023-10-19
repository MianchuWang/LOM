import torch
import numpy as np

from agents.td3bc import TD3BC, BaseAgent
from networks.networks import Policy, Qnetwork, Binary_Classifier, Dynamics, Vnetwork

class SOTA(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        
        self.training_steps = 0
        self.beta = 2
        self.policy_delay = 2
        self.I = 1
        
        self.policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=5e-5)
        self.v_nets, self.v_target_nets, self.v_net_opts = [], [], []
        for i in range(self.I):
            self.v_nets.append(Vnetwork(self.state_dim).to(device=self.device))
            self.v_target_nets.append(Vnetwork(self.state_dim).to(device=self.device))
            self.v_target_nets[i].load_state_dict(self.v_nets[i].state_dict())
            self.v_net_opts.append(torch.optim.Adam(self.v_nets[i].parameters(), lr=1e-4))  
        
        
        self.policy_info = {}
        self.value_info = {}
        
        
    def train_models(self):
        self.value_info = self.train_value_function(batch_size=256)
        if self.training_steps % self.policy_delay == 0:
            self.train_improved_value_function(batch_size=256)
            self.policy_info = self.train_policy(batch_size=256)
            self.update_target_nets(self.v_nets, self.v_target_nets)
        self.training_steps += 1
        return {**self.value_info, **self.policy_info}
    
    def train_value_function(self, batch_size):
        for i in range(self.I):
            states, _, _, _, _, returns  = self.replay_buffer.sample_with_returns(batch_size)
            states_prep, _, _, _, _, = self.preprocess(states=states)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            pred_v_value = self.v_nets[i](states_prep)
            v_loss = ((pred_v_value - returns)**2).mean()
            
            self.v_net_opts[i].zero_grad()
            v_loss.backward()
            self.v_net_opts[i].step()
        
        return {'V/loss': v_loss.item()}
    
    def train_improved_value_function(self, batch_size):
        for i in range(self.I):
            states, _, _, _, _, returns  = self.replay_buffer.sample_with_returns(batch_size)
            states_prep, _, _, _, _, = self.preprocess(states=states)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            pred_v_value = self.v_nets[i](states_prep)
            v_loss = (torch.clip(pred_v_value - returns, 0, 10000)**2).mean()
            
            self.v_net_opts[i].zero_grad()
            v_loss.backward()
            self.v_net_opts[i].step()
        
        return {'V/loss': v_loss.item()}
        
    def train_policy(self, batch_size):
        states, actions, rewards, next_states, terminals, returns  = self.replay_buffer.sample_with_returns(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        _, gen_actions = self.policy(states_prep)
        with torch.no_grad():
            values = self.v_nets[0](states_prep)
            advantage = returns - values
            exp_adv = torch.clip(torch.exp(self.beta * advantage), -10000, 20)
            weights = exp_adv.squeeze(dim=1)
            
        policy_loss = ((gen_actions - actions_prep.detach()).pow(2).mean(dim=1) * weights).mean()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        return {'policy/loss': policy_loss.item(),
                'policy/weights': weights.mean().item()}
    
    def get_action(self, state):
        with torch.no_grad():
            state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
            _, action = self.policy(state_prep)
        return action.cpu().numpy().squeeze()
    
    def update_target_nets(self, net, target_net):
        for k in range(len(net)):
            for param, target_param in zip(net[k].parameters(), target_net[k].parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
