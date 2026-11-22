import unittest

import numpy as np
import torch

from agents.lom import LOM


class SyntheticReplayBuffer:
    def __init__(self, size=128, state_dim=4, action_dim=2):
        rng = np.random.default_rng(3)
        self.obs = rng.normal(size=(size, state_dim)).astype(np.float32)
        self.actions = np.tanh(
            rng.normal(size=(size, action_dim))).astype(np.float32)
        self.next_obs = np.roll(self.obs, -1, axis=0)
        self.rewards = rng.normal(size=(size, 1)).astype(np.float32)
        self.terminals = np.zeros((size, 1), dtype=np.float32)

    def get_bounds(self):
        return len(self.obs)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.obs), batch_size)
        return (self.obs[indices], self.actions[indices], self.rewards[indices],
                self.next_obs[indices], self.terminals[indices])

    def sample_with_next_action(self, batch_size):
        indices = np.random.randint(0, len(self.obs) - 1, batch_size)
        return (self.obs[indices], self.actions[indices], self.rewards[indices],
                self.next_obs[indices], self.actions[indices + 1],
                self.terminals[indices])


class LomIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_agent(self, mode_selection):
        return LOM(
            env_name='halfcheetah-medium-replay-v2',
            replay_buffer=SyntheticReplayBuffer(), state_dim=4, ac_dim=2,
            device='cpu', discount=0.99, normalise=1,
            mode_selection=mode_selection, top_k=3, mode_action_samples=2,
            gnn_hidden_dim=16, gnn_lr=3e-4, gnn_ranking_weight=0.25,
            gmm_pretrain_steps=0, gmm_batch_size=8, gmm_checkpoint='',
            save_gmm=0,
        )

    def test_all_three_configurations_train_one_step(self):
        for mode_selection in ('top1', 'topk_q', 'topk_gnn', 'topk_gnn_merge'):
            with self.subTest(mode_selection=mode_selection):
                agent = self.make_agent(mode_selection)
                info = agent.train_models()
                action = agent.get_action(np.zeros(4, dtype=np.float32))
                self.assertEqual(action.shape, (2,))
                self.assertTrue(np.isfinite(action).all())
                self.assertTrue(all(np.isfinite(value) for value in info.values()))
                self.assertEqual('gnn/loss' in info,
                                 mode_selection in ('topk_gnn', 'topk_gnn_merge'))
                if mode_selection == 'topk_gnn_merge':
                    self.assertTrue(np.isfinite(info['gnn/merge_entropy']))
                    self.assertTrue(np.isfinite(info['gnn/effective_modes']))


if __name__ == '__main__':
    unittest.main()
