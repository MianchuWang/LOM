import unittest

import torch
import torch.nn as nn

from agents.lom import LOM


class FixedModeNetwork(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.register_buffer('scores', torch.tensor(scores, dtype=torch.float32))

    def forward(self, states, one_hot_components):
        component = one_hot_components.argmax(dim=1)
        return self.scores[component].unsqueeze(1)


class FixedGMM(nn.Module):
    def __init__(self, num_mixtures, action_dim):
        super().__init__()
        means = torch.arange(num_mixtures, dtype=torch.float32)
        self.register_buffer('means', means[:, None].expand(-1, action_dim) / 10)
        self.register_buffer('weights', torch.arange(1, num_mixtures + 1, dtype=torch.float32))

    def forward(self, states):
        batch_size = states.shape[0]
        means = self.means.unsqueeze(0).expand(batch_size, -1, -1)
        log_stds = torch.full_like(means, -2.0)
        weights = self.weights / self.weights.sum()
        return means, log_stds, weights.unsqueeze(0).expand(batch_size, -1)


class LomTopKTest(unittest.TestCase):
    def make_agent(self):
        agent = LOM.__new__(LOM)
        agent.device = torch.device('cpu')
        agent.state_dim = 4
        agent.ac_dim = 2
        agent.params = {'num_mixtures': 5}
        agent.top_k = 3
        agent.mode_net = FixedModeNetwork([0.1, 0.8, -0.2, 0.4, 1.2])
        agent.gmm = FixedGMM(5, 2)
        return agent

    def test_topk_is_ranked_and_parameters_follow_indices(self):
        agent = self.make_agent()
        modes = agent.get_topk_modes(torch.randn(2, 4))
        expected_indices = torch.tensor([[4, 1, 3], [4, 1, 3]])
        self.assertTrue(torch.equal(modes['indices'], expected_indices))
        self.assertTrue(torch.allclose(
            modes['scores'][0], torch.tensor([1.2, 0.8, 0.4])))
        self.assertTrue(torch.allclose(
            modes['means'][0, :, 0], torch.tensor([0.4, 0.1, 0.3])))
        self.assertEqual(modes['stds'].shape, (2, 3, 2))
        self.assertEqual(modes['weights'].shape, (2, 3))

    def test_gather_supports_scalar_and_vector_mode_values(self):
        indices = torch.tensor([[2, 0]])
        scalars = torch.tensor([[10.0, 20.0, 30.0]])
        vectors = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        self.assertTrue(torch.equal(
            LOM._gather_modes(scalars, indices), torch.tensor([[30.0, 10.0]])))
        self.assertTrue(torch.equal(
            LOM._gather_modes(vectors, indices),
            torch.tensor([[[5.0, 6.0], [1.0, 2.0]]])))

    def test_topk_selection_edge_cases(self):
        agent = self.make_agent()
        for k, expected in ((1, [4]), (5, [4, 1, 3, 0, 2])):
            with self.subTest(k=k):
                agent.top_k = k
                modes = agent.get_topk_modes(torch.randn(1, 4))
                self.assertEqual(modes['indices'].tolist(), [expected])
                self.assertEqual(modes['means'].shape, (1, k, 2))


if __name__ == '__main__':
    unittest.main()
