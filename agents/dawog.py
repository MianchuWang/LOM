import torch
import numpy as np
import torch.nn as nn

from networks.networks import Policy, v_network
from agents.base_agent import BaseAgent

class Partition:
    def __init__(self, K, v_net):
        self.K = K
        self.v_net = v_net
        self.xi_0 = 0
        self.xi_K = 1
    def get_region(self, states, goals):
        states = torch.Tensor(states).cuda()
        goals = torch.Tensor(goals).cuda()
        
        values = self.v_net[0](states, goals)
        for i in range(1, len(self.v_net)):
            values += self.v_net[i](states, goals)
        values = values / len(self.v_net)
        
        values = values.cpu().numpy()
        values = np.clip(values, self.xi_0, self.xi_K)
        region = np.floor(((self.K-1) * ((values - self.xi_0) / (self.xi_K - self.xi_0))))
        return region.squeeze()

class v_partition_network(nn.Module):
    def __init__(self, state_dim, goal_dim, n_partition):
        super(v_partition_network, self).__init__()
        self.state_dim = state_dim
        self.model = nn.Sequential(nn.Linear(state_dim + goal_dim + n_partition, 256), 
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
    def forward(self, state, goal, partition):
        input = torch.cat([state, goal, partition], dim=1)
        output = self.model(input)
        return output
        
class DAWOG(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.num_regions = 20
        self.beta1 = 25
        self.beta2 = 25
        self.her_prob = 0.0
        self.training_steps = 0
        self.I = 1
        
        
        self.policy = Policy(self.state_dim, self.ac_dim, self.goal_dim).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3) 
        
        self.v_net = [v_network(self.state_dim, self.goal_dim).to(device=self.device) for i in range(self.I)]
        self.v_target_net = [v_network(self.state_dim, self.goal_dim).to(device=self.device) for i in range(self.I)]
        for i in range(self.I):
            self.v_target_net[i].load_state_dict(self.v_net[i].state_dict())
            self.v_net_opt = torch.optim.Adam(self.v_net[i].parameters(), lr=1e-3)
        
        self.v_partition_net = v_partition_network(self.state_dim, self.goal_dim, self.num_regions).to(device=self.device)
        self.v_partition_target_net = v_partition_network(self.state_dim, self.goal_dim, self.num_regions).to(device=self.device)
        self.v_partition_target_net.load_state_dict(self.v_partition_net.state_dict())
        self.v_partition_net_opt = torch.optim.Adam(self.v_partition_net.parameters(), lr=1e-3)
        
        self.partition = Partition(self.num_regions, self.v_net)
    
    def train_models(self):
        value_info, partition_value_info = {}, {}
        if self.beta1 > 0: partition_value_info = self.train_partition_value_function(batch_size=512)
        if self.beta2 > 0: value_info = self.train_value_function(batch_size=512)
        if self.training_steps % 2 == 0:
            self.update_target_nets(self.v_net, self.v_target_net)
            self.update_target_nets([self.v_partition_net], [self.v_partition_target_net])
        self.training_steps += 1
        policy_info = self.train_policy(batch_size=512)
        return {**value_info, **partition_value_info, **policy_info}
    
    @torch.no_grad()
    def get_mean_value(self, states, goals):
        values = self.v_net[0](states, goals)
        for i in range(1, self.I):
            values += self.v_net[i](states, goals)
        values = values / self.I
        return values
    
    def train_policy(self, batch_size): 
        states, actions, next_states, goals, _ = self.replay_buffer.sample(batch_size, her_prob=self.her_prob)
        states_prep, actions_prep, next_states_prep, goals_prep = \
            self.preprocess(states=states, actions=actions, next_states=next_states, goals=goals)
        achieved_goals = self.get_goal_from_state(next_states)
        rewards = self.compute_reward(achieved_goals, goals, None)[..., np.newaxis]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            c_t = self.partition.get_region(states, goals)
            c_next = self.partition.get_region(next_states, goals)
            expected_c_next = np.minimum(c_t + 1, np.zeros_like(c_t) + self.num_regions - 1)
            partition_rewards = c_next == expected_c_next
            c_next_oh = self.to_one_hot(expected_c_next.astype(np.int32))
            
            partition_rewards_tensor = torch.tensor(partition_rewards.astype(np.int32)[..., np.newaxis], dtype=torch.float32, device=self.device)
            pred_v_partition_value = self.v_partition_net(states_prep, goals_prep, c_next_oh)
            next_v_partition_value = self.v_partition_net(next_states_prep, goals_prep, c_next_oh)
            A_partition = partition_rewards_tensor + (1-partition_rewards_tensor) * self.discount * next_v_partition_value - pred_v_partition_value 
            
            pred_v_value = self.get_mean_value(states_prep, goals_prep)
            next_v_value = self.get_mean_value(next_states_prep, goals_prep)
            A = rewards_tensor + (1-rewards_tensor) * self.discount * next_v_value - pred_v_value
            
            weights = torch.clamp(torch.exp(self.beta1 * A_partition + self.beta2 *  A), 0, 10)
            
        ac_dist, ac_mean = self.policy(states_prep, goals_prep)
        log_prob = ac_dist.log_prob(actions_prep)
        policy_loss = - (log_prob.mean(dim=1) * weights.squeeze()).mean()
      
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        return {'policy_loss': policy_loss.item()}
    
    def train_partition_value_function(self, batch_size):
        states, actions, next_states, goals, _ = self.replay_buffer.sample(batch_size, her_prob=self.her_prob)
        with torch.no_grad():
            c_t = self.partition.get_region(next_states, goals)
            c_g = np.ones_like(c_t) * (self.num_regions - 1)
            rewards = c_t == c_g
            c_g = self.to_one_hot(c_g.astype(np.int32))
        
            rewards_tensor = torch.tensor(rewards.astype(np.int32)[..., np.newaxis], dtype=torch.float32, device=self.device)
            states_prep, actions_prep, next_states_prep, goals_prep = \
                self.preprocess(states=states, actions=actions, next_states=next_states, goals=goals)
                
            v_next_value = self.v_partition_target_net(next_states_prep, goals_prep, c_g)
            target_v_value = rewards_tensor + (1-rewards_tensor) * self.discount * v_next_value
        pred_v_value = self.v_partition_net(states_prep, goals_prep, c_g)
        v_loss = ((target_v_value - pred_v_value)**2).mean()
        self.v_partition_net_opt.zero_grad()
        v_loss.backward()
        self.v_partition_net_opt.step()
        
        return {'v_partition_loss': v_loss.item(), 
                'partition_pred_value': pred_v_value.mean().item(), 
                'partition_target_value': target_v_value.mean().item()}
    
    def train_value_function(self, batch_size):
        for i in range(self.I):
            states, actions, next_states, goals, _ = self.replay_buffer.sample(batch_size, her_prob=self.her_prob)
            achieved_goals = self.get_goal_from_state(next_states)
            rewards = self.compute_reward(achieved_goals, goals, None)[..., np.newaxis]
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device) + 1
            states_prep, actions_prep, next_states_prep, goals_prep = \
                self.preprocess(states=states, actions=actions, next_states=next_states, goals=goals)
            with torch.no_grad():
                v_next_value = self.v_target_net[i](next_states_prep, goals_prep)
                target_v_value = rewards_tensor + (1-rewards_tensor) * self.discount * v_next_value
            pred_v_value = self.v_net[i](states_prep, goals_prep)
            v_loss = ((target_v_value - pred_v_value)**2).mean()
            self.v_net_opt.zero_grad()
            v_loss.backward()
            self.v_net_opt.step()
        
        return {'v_loss': v_loss.item(), 
                'pred_value': pred_v_value.mean().item(), 
                'target_value': target_v_value.mean().item()}
    
    def update_target_nets(self, nets, target_nets):
        for net, target_net in zip(nets, target_nets):
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
    
    def get_action(self, state, goal):
        with torch.no_grad():
            state_prep, _, _, goal_prep = \
                self.preprocess(states=state[np.newaxis], goals=goal[np.newaxis])
            ac_dist, action = self.policy(state_prep, goal_prep)
        return ac_dist.sample().cpu().numpy().squeeze()
        
    def to_one_hot(self, array, return_tensor=True):
        one_hot_array = np.eye(self.num_regions)[array]
        if return_tensor:
            one_hot_array = torch.tensor(one_hot_array, dtype=torch.float32, device=self.device)
        return one_hot_array
    
        
            
            
            
            
            
            
            
            
            
            
            
            
        
