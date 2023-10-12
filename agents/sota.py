import torch
import numpy as np

from agents.td3bc import TD3BC 
from networks.networks import Policy, Qnetwork, Binary_Classifier, Dynamics

class SOTA(TD3BC):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.classifier = Binary_Classifier(self.state_dim, self.ac_dim).to(device=self.device)
        self.classifier_opt = torch.optim.Adam(self.classifier.parameters(), lr=3e-4)
        self.dynamics = Dynamics(self.state_dim, self.ac_dim).to(device=self.device)
        self.dynamics_opt = torch.optim.Adam(self.dynamics.parameters(), lr=3e-4)
        
        #self.train_behaviour_policy(batch_size=256, steps=50000)
        self.policy_info = {}
        self.classifier_info = {}
        self.value_info = {}
        self.dynamics_info = {}
    
    def train_models(self):
        self.value_info = self.train_value_function(batch_size=256)
        if self.training_steps % self.policy_delay == 0:
            self.policy_info = self.train_policy(batch_size=256)
            self.classifier_info = self.train_binary_classifier(batch_size=256)
            #self.dynamics_info = self.train_dynamics(batch_size=256)
            
            self.update_target_nets(self.q_nets, self.q_target_nets)
            self.update_target_nets([self.policy], [self.target_policy])
        self.training_steps += 1
        return {**self.value_info, **self.policy_info, **self.classifier_info,
                **self.dynamics_info}
    
    def train_dynamics(self, batch_size):
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size=batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, _ = self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states)
        pred_next_states = self.dynamics(states_prep, actions_prep)
        dynamics_loss = (pred_next_states - next_states_prep).pow(2).mean()
        self.dynamics_opt.zero_grad()
        dynamics_loss.backward()
        self.dynamics_opt.step()
        return {'dynamics/loss': dynamics_loss.item()}
    
    
    def train_binary_classifier(self, batch_size):
        states, actions, _, _, _ = self.replay_buffer.sample(batch_size=batch_size)
        states_prep, actions_prep, _, _, _ = self.preprocess(states=states, actions=actions)
        _, actions_fake_prep = self.policy(states_prep)
        
        input_ac = torch.cat([actions_prep, actions_fake_prep], dim=0)
        input_state = torch.cat([states_prep, states_prep], dim=0)
        prob = self.classifier(input_state, input_ac)
        
        labels = torch.cat([torch.ones(actions_prep.shape[0]), 
                            torch.zeros(actions_fake_prep.shape[0])]).unsqueeze(-1).to(self.device)
        criterion = torch.nn.BCELoss()
        classifier_loss = criterion(prob, labels)
        
        self.classifier_opt.zero_grad()
        classifier_loss.backward()
        self.classifier_opt.step()
        return {'classifier/loss': classifier_loss.item()}
        
        
    def train_policy(self, batch_size):
        states, _, _, _, _  = self.replay_buffer.sample(batch_size)
        states_prep, _, _, _, _ = self.preprocess(states=states)
        
        gen_dist, gen_actions = self.policy(states_prep)
        actions_prep = gen_dist.rsample()
        with torch.no_grad():
            # The exponential-advantage weight
            q_values = self.q_nets[0](states_prep, actions_prep)
            curr_q_values = self.q_nets[0](states_prep, gen_actions)
            advs = q_values - curr_q_values
            weights_exp = torch.clip(torch.exp(2 * advs), -10000, 100).squeeze()
            
            weights = weights_exp
        
        main_loss = - (gen_dist.log_prob(actions_prep.detach()).mean(dim=1) * weights).mean()
        reg_loss = - self.classifier(states_prep, actions_prep).mean()
        policy_loss = main_loss + 10 * reg_loss
        
        states_prep = self.dynamics(states_prep, actions_prep).detach()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        return {'policy/loss': policy_loss.item(),
                'policy/weights': weights.mean().item(),
                'policy/main_loss': main_loss.item(),
                'policy/reg_loss': reg_loss.item()}
