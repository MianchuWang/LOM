import torch
import numpy as np

from agents.base_agent import BaseAgent
from networks.networks import Policy, Qnetwork

class STR(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        
        self.behaviour_policy = Policy(self.state_dim, self.ac_dim).to(device=self.device)
        self.behaviour_policy_opt = torch.optim.Adam(self.behaviour_policy.parameters(), lr=3e-4)
        
        self.q_net1 = Qnetwork(self.state_dim, self.ac_dim).to(device=self.device)
        self.q_target_net1 = Qnetwork(self.state_dim, self.ac_dim).to(device=self.device)
        self.q_target_net1.load_state_dict(self.q_net1.state_dict())
        self.q_net_opt1 = torch.optim.Adam(self.q_net1.parameters(), lr=3e-4)
        
        self.q_net2 = Qnetwork(self.state_dim, self.ac_dim).to(device=self.device)
        self.q_target_net2 = Qnetwork(self.state_dim, self.ac_dim).to(device=self.device)
        self.q_target_net2.load_state_dict(self.q_net2.state_dict())
        self.q_net_opt2 = torch.optim.Adam(self.q_net2.parameters(), lr=3e-4)
        
        self.policy_noise = 0.1
        self.noise_clip = 0.1
        self.training_steps = 0
        self.max_action = 1
        

    def train_models(self, batch_size=256):
        if self.training_steps == 0:
            print('Training the Gaussian density estimator ...')
            for _ in range(10000):
                states, actions, _, _, _ = self.replay_buffer.sample(batch_size)
                states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
                ac_dist, ac_mean = self.behaviour_policy(states_prep)
                log_prob = ac_dist.log_prob(actions_prep)
                behaviour_policy_loss = - log_prob.mean()
                self.behaviour_policy_opt.zero_grad()
                behaviour_policy_loss.backward()
                self.behaviour_policy_opt.step()
            self.policy.load_state_dict(self.behaviour_policy.state_dict())
        
        value_info = self.train_value_function(batch_size=batch_size)
        policy_info = self.train_policy(batch_size=batch_size)
        if self.training_steps % 2 == 0:
            self.update_target_nets(self.q_net, self.q_target_net)
        self.training_steps += 1
        
        return {**value_info, **policy_info}
    
    
    def train_value_function(self, batch_size):
        states, actions, rewards, next_states, terminals  = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, 
                            next_states=next_states, terminals=terminals)
        
        
        
        q1, q2 = self.q_net1(states_prep, actions_prep), self.q_net2(states_prep, actions_prep)
        with torch.no_grad():
            noise = (torch.randn_like(actions_prep) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.policy(next_states_prep) + noise).clamp(-self.max_action, self.max_action)
            next_q = torch.min(self.q_target_net1(next_states_prep, next_actions), 
                               self.q_target_net2(next_states_prep, next_actions))
            target_q = rewards_prep + self.discount * (1 - terminals_prep) * next_q
        
        q_net1_loss = ((q1 - target_q).pow(2)).mean()
        q_net2_loss = ((q2 - target_q).pow(2)).mean()

        self.q_net_opt1.zero_grad()
        q_net1_loss.backward()
        self.q_net_opt1.step()

        self.q_net_opt2.zero_grad()
        q_net2_loss.backward()
        self.q_net_opt2.step()
        
        return {'q_loss': q_loss.item(), 
                'pred_value': pred_value.mean().item(), 
                'target_value': target_value.mean().item()}
    
    def train_policy(self, batch_size):

        states, actions, rewards, next_states, terminals  = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, 
                            next_states=next_states, terminals=terminals)
        with torch.no_grad():
            value = self.q_net(states_prep, actions_prep)
            _, next_actions_prep = self.policy(next_states_prep)    
            next_value = self.q_net(next_states_prep, next_actions_prep)
            A = value - next_value
            weights = torch.clamp(torch.exp(A), 0, 10)
        gen_dist, gen_actions = self.policy(states_prep)
        policy_loss = (gen_dist.log_prob(actions_prep).mean(dim=1) * weights.squeeze()).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        return {'policy_loss': policy_loss.item()}
        
        

    def get_action(self, state):
        with torch.no_grad():
            state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
            ac_dist, action = self.policy(state_prep)
        return action.cpu().numpy().squeeze()
    
    def update_target_nets(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
