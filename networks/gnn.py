import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingLayer(nn.Module):
    """A small fully-connected message-passing layer with distance edges."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, nodes, edge_distances):
        batch_size, num_nodes, _ = nodes.shape
        receiver = nodes.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        sender = nodes.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        messages = self.message(
            torch.cat([receiver, sender, edge_distances.unsqueeze(-1)], dim=-1)
        )

        if num_nodes == 1:
            aggregated = torch.zeros_like(nodes)
        else:
            mask = ~torch.eye(num_nodes, dtype=torch.bool, device=nodes.device)
            mask = mask.view(1, num_nodes, num_nodes, 1)
            aggregated = (messages * mask).sum(dim=2) / (num_nodes - 1)

        return nodes + self.update(torch.cat([nodes, aggregated], dim=-1))


class ModeGraphNetwork(nn.Module):
    """Refine scores for selected Gaussian modes using their relationships.

    Node features contain mean, standard deviation, mixture weight, and the
    original Hyper-Q score. Euclidean distances between means are edge
    features in a fully-connected graph.
    """

    def __init__(self, node_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden_dim) for _ in range(num_layers)
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, means):
        if node_features.ndim != 3 or means.ndim != 3:
            raise ValueError("node_features and means must have shape [batch, nodes, features]")
        if node_features.shape[:2] != means.shape[:2]:
            raise ValueError("node_features and means must use the same batch and node dimensions")

        edge_distances = torch.cdist(means, means, p=2)
        nodes = self.node_encoder(node_features)
        for layer in self.layers:
            nodes = layer(nodes, edge_distances)
        return self.score_head(nodes).squeeze(-1)


def mode_ranking_loss(predictions, targets, ranking_weight=0.25):
    """Regress normalized values and preserve their within-state ordering."""
    regression = F.smooth_l1_loss(predictions, targets)
    if predictions.shape[1] < 2:
        return regression, regression, predictions.new_zeros(())

    pred_differences = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    target_differences = targets.unsqueeze(2) - targets.unsqueeze(1)
    pair_mask = target_differences.abs() > 1e-6
    if not pair_mask.any():
        ranking = predictions.new_zeros(())
    else:
        labels = (target_differences[pair_mask] > 0).to(predictions.dtype)
        ranking = F.binary_cross_entropy_with_logits(
            pred_differences[pair_mask], labels
        )
    total = regression + ranking_weight * ranking
    return total, regression, ranking
