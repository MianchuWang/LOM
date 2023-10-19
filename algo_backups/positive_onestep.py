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
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.v_nets, self.v_target_nets, self.v_net_opts = [], [], []
        for i in range(self.I):
            self.v_nets.append(Vnetwork(self.state_dim).to(device=self.device))
            self.v_target_nets.append(Vnetwork(self.state_dim).to(device=self.device))
            self.v_target_nets[i].load_state_dict(self.v_nets[i].state_dict())
            self.v_net_opts.append(torch.optim.Adam(self.v_nets[i].parameters(), lr=3e-4))  
        
        
        self.policy_info = {}
        self.classifier_info = {}
        self.value_info = {}
        self.dynamics_info = {}
        
        
    def train_models(self):
        self.value_info = self.train_value_function(batch_size=256)
        if self.training_steps % self.policy_delay == 0:
            self.policy_info = self.train_policy(batch_size=256)
            
            self.update_target_nets(self.v_nets, self.v_target_nets)
        self.training_steps += 1
        return {**self.value_info, **self.policy_info, **self.classifier_info,
                **self.dynamics_info}
    
    def train_value_function(self, batch_size):
        for i in range(self.I):
            states, _, rewards, next_states, terminals  = self.replay_buffer.sample(batch_size)
            states_prep, _, rewards_prep, next_states_prep, terminals_prep = self.preprocess(states=states, rewards=rewards, next_states=next_states, terminals=terminals)
            
            with torch.no_grad():
                v_next_value = self.v_target_nets[i](next_states_prep)
                target_v_value = rewards_prep + (1-terminals_prep) * self.discount * v_next_value
            pred_v_value = self.v_nets[i](states_prep)
            v_loss = ((target_v_value - pred_v_value)**2).mean()
            self.v_net_opts[i].zero_grad()
            v_loss.backward()
            self.v_net_opts[i].step()
        
        return {'V/loss': v_loss.item(), 
                'V/value': pred_v_value.mean().item(), 
                'V/target_value': target_v_value.mean().item()}
        
    def train_policy(self, batch_size):
        states, actions, rewards, next_states, terminals, returns  = self.replay_buffer.sample_with_returns(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        _, gen_actions = self.policy(states_prep)
        with torch.no_grad():
            values = self.v_nets[0](states_prep)
            advantage = returns - values
            is_positive = (advantage > -10000).to(dtype=torch.float)
            exp_adv = torch.clip(torch.exp(self.beta * advantage), -10000, 100)
            weights = exp_adv.squeeze(dim=1) * is_positive.squeeze(dim=1)
            
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
