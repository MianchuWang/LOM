import torch
import numpy as np

from agents.base_agent import BaseAgent
from networks.networks import Policy
from networks.networks import Generator, Discriminator, Vnetwork

class WCGAN(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.noise_dim = 10
        self.beta = 2
        self.M = 50
        self.generator = Generator(self.state_dim, self.ac_dim, self.noise_dim).to(device=self.device)
        self.discriminator = Discriminator(self.state_dim, self.ac_dim).to(device=self.device)
        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.generator.eval()
        self.discriminator.eval()
        
        self.v_net = Vnetwork(self.state_dim).to(device=self.device)
        self.v_target_net = Vnetwork(self.state_dim).to(device=self.device)
        self.v_target_net.load_state_dict(self.v_net.state_dict())
        self.v_net_opt = torch.optim.Adam(self.v_net.parameters(), lr=1e-3)
        
        self.training_steps = 0

    def train_models(self, batch_size=512):
        gan_info = self.train_gan(batch_size=512)
        value_function_info = self.train_value_function(batch_size=512)
        if self.training_steps % 2 == 0:
            self.update_target_nets(self.v_net, self.v_target_net)
        self.training_steps += 1
        return {**gan_info, **value_function_info}
    
    def train_value_function(self, batch_size):
        states, actions, rewards, next_states, terminals  = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        with torch.no_grad():
            v_next_value = self.v_target_net(next_states_prep)
            target_v_value = rewards_prep + (1-terminals_prep) * self.discount * v_next_value
        pred_v_value = self.v_net(states_prep)
        v_loss = ((target_v_value - pred_v_value) ** 2).mean()
        self.v_net_opt.zero_grad()
        v_loss.backward()
        self.v_net_opt.step()

        return {'v_loss': v_loss.item(),
                'v_value': pred_v_value.mean().item(), 
                'v_value_std': pred_v_value.std().item()}
    
    def get_action(self, state):
        with torch.no_grad():
            state_prep, _, _, _, _ = self.preprocess(states=state[np.newaxis])
            noise = torch.zeros(1, self.noise_dim, device=self.device)
            action = self.generator(state_prep, noise)
        return action.cpu().numpy().squeeze()
    
    def train_gan(self, batch_size):
        states, actions, rewards, next_states, terminals  = self.replay_buffer.sample(batch_size)
        states_prep, actions_prep, rewards_prep, next_states_prep, terminals_prep = \
            self.preprocess(states=states, actions=actions, rewards=rewards, next_states=next_states, terminals=terminals)
        # Real actions -> higher score
        # Fake actions -> lower score
        states_prep += 0.0001 * torch.randn(*states_prep.shape, device=self.device)
        next_states_prep += 0.0001 * torch.randn(*states_prep.shape, device=self.device)
        self.discriminator.train()
        self.generator.train()
        
        pred_v_value = self.v_net(states_prep)
        next_v_value = self.v_net(next_states_prep)
        A = rewards_prep + (1 - terminals_prep) * self.discount * next_v_value - pred_v_value
        clip_exp_A = torch.clamp(torch.exp(self.beta * A), 0, self.M)
        weights = clip_exp_A

        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_actions = self.generator(states_prep, noise)
        dis_loss_fake = self.discriminator(states_prep, fake_actions)
        dis_loss_real = self.discriminator(states_prep, actions_prep)
        dis_loss = - (weights * torch.log(dis_loss_real)).mean() - weights.mean() * torch.log(1 - dis_loss_fake).mean()
        self.discriminator_opt.zero_grad()
        dis_loss.backward()
        self.discriminator_opt.step()
        
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_actions = self.generator(states_prep, noise)
        gen_loss = torch.log(1 - self.discriminator(states_prep, fake_actions)).mean()
        self.generator_opt.zero_grad()
        gen_loss.backward()
        self.generator_opt.step()

        self.generator.eval()
        self.discriminator.eval()
        return {'advantages': A,
                'clip_exp_A': clip_exp_A,
                'weights': weights,
                'generator_loss': gen_loss.item(),
                'discriminator_loss': dis_loss.item()}
    
    def update_target_nets(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
