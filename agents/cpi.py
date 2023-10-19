import torch
from agents.str import STR

class CPI(STR):
    def __init__(self, **agent_params):
        super().__init__(bc_initialization=0, **agent_params)
        
    def train_models(self):
        if self.training_steps % 2 == 0:
            self.value_info = self.train_value_function(batch_size=256)
        else:
            self.value_info = self.train_improved_value_function(batch_size=256)
        if self.training_steps % self.policy_delay == 0:
            self.policy_info = self.train_policy(batch_size=256)
            self.update_target_nets(self.q_nets, self.q_target_nets)
            self.update_target_nets([self.policy], [self.target_policy])
        self.training_steps += 1
        return {**self.value_info, **self.policy_info}
        
    def train_improved_value_function(self, batch_size):
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
            main_loss = ((target_q_value - pred_q_value)**2).squeeze()
            
            with torch.no_grad():
                _, curr_actions = self.target_policy(states_prep)
                weights = self.q_target_nets[k](states_prep, actions_prep) - self.q_target_nets[k](states_prep, curr_actions)
                # CPI-exp
                # weights = (weights.exp() / weights.exp().sum()).squeeze()
                # CPI-pos
                # weights = (weights > 0).to(dtype=torch.float).squeeze()
            q_loss = (main_loss * weights).mean()
                
            self.q_net_opts[k].zero_grad()
            q_loss.backward()
            self.q_net_opts[k].step()
        
        return {'Q/loss': q_loss.item(), 
                'Q/pred_value': pred_q_value.mean().item(), 
                'Q/target_value': target_q_value.mean().item(),
                'Q/weights': weights.mean().item()}

       