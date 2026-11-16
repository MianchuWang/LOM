import unittest

import torch

from networks.gnn import ModeGraphNetwork, mode_ranking_loss


class ModeGraphNetworkTest(unittest.TestCase):
    def test_output_shape_and_finite_scores(self):
        model = ModeGraphNetwork(node_dim=8, hidden_dim=16, num_layers=2)
        features = torch.randn(4, 3, 8)
        means = torch.randn(4, 3, 3)
        scores = model(features, means)
        self.assertEqual(scores.shape, (4, 3))
        self.assertTrue(torch.isfinite(scores).all())

    def test_k_edge_cases(self):
        model = ModeGraphNetwork(node_dim=6, hidden_dim=8, num_layers=2)
        for k in (1, 5):
            scores = model(torch.randn(2, k, 6), torch.randn(2, k, 2))
            self.assertEqual(scores.shape, (2, k))
            self.assertTrue(torch.isfinite(scores).all())

    def test_distance_edges_are_symmetric(self):
        means = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [0.0, 4.0]]])
        distances = torch.cdist(means, means)
        self.assertTrue(torch.allclose(distances, distances.transpose(1, 2)))
        self.assertTrue(torch.allclose(distances[0, 0], torch.tensor([0.0, 5.0, 4.0])))

    def test_loss_decreases_and_updates_parameters(self):
        torch.manual_seed(7)
        model = ModeGraphNetwork(node_dim=6, hidden_dim=16, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        features = torch.randn(16, 3, 6)
        means = features[..., :2]
        targets = means[..., 0] - 0.5 * means[..., 1]
        targets = (targets - targets.mean(1, keepdim=True)) / targets.std(1, keepdim=True)
        initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]

        losses = []
        for _ in range(80):
            predictions = model(features, means)
            loss, _, _ = mode_ranking_loss(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0] * 0.5)
        self.assertTrue(any(
            not torch.allclose(before, after)
            for before, after in zip(initial_parameters, model.parameters())
        ))


if __name__ == '__main__':
    unittest.main()
