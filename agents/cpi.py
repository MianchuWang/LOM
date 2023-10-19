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
        self.policy_info = self.train_policy(batch_size=256)
        if self.training_steps % self.policy_delay == 0:
            self.update_target_nets(self.q_nets, self.q_target_nets)
        self.training_steps += 1
        return {**self.value_info, **self.policy_info}
        
    def train_improved_value_function(self, batch_size):
        for i in range(self.K):
            states, actions, _, _, _, returns  = self.replay_buffer.sample_with_returns(batch_size)
            states_prep, actions_prep, _, _, _, = self.preprocess(states=states, actions=actions)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            pred_values = self.q_nets[i](states_prep, actions_prep)
            pmr = torch.clip(returns - pred_values, 0, 1e5)
            q_loss = (pmr**2).mean()
            
            self.q_net_opts[i].zero_grad()
            q_loss.backward()
            self.q_net_opts[i].step()
        
        return {'Q/loss': q_loss.item()}


       