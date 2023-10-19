import torch
from agents.awr import AWR

class BPI(AWR):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        
    def train_models(self):
        if self.training_steps % 2 == 0:
            self.value_info = self.train_value_function(batch_size=256)
        else:
            self.value_info = self.train_improved_value_function(batch_size=256)
        self.policy_info = self.train_policy(batch_size=256)
        if self.training_steps % self.policy_delay == 0:
            self.update_target_nets(self.v_nets, self.v_target_nets)
        self.training_steps += 1
        return {**self.value_info, **self.policy_info, **self.classifier_info,
                **self.dynamics_info}
        
    def train_improved_value_function(self, batch_size):
        for i in range(self.I):
            states, _, _, _, _, returns  = self.replay_buffer.sample_with_returns(batch_size)
            states_prep, _, _, _, _, = self.preprocess(states=states)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            pred_values = self.v_nets[i](states_prep)
            pmr = torch.clip(returns - pred_values, 0, 1e5)
            v_loss = (pmr**2).mean()
            
            self.v_net_opts[i].zero_grad()
            v_loss.backward()
            self.v_net_opts[i].step()
        
        return {'V/loss': v_loss.item(),
                'V/target_value': pred_values.mean().item()}

    