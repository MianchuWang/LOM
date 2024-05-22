def mixture_gaussian_loss(policy, states, actions):
    means, log_stds, weights = policy(states)
    stds = log_stds.exp()

    # Calculate the log probabilities of the actions under each Gaussian component
    m = torch.distributions.Normal(means, stds)
    log_probs = m.log_prob(actions.unsqueeze(1).expand_as(means))

    # Sum log probabilities over action dimensions
    log_probs = log_probs.sum(-1)

    # Weight the log probabilities by the mixture weights
    weighted_log_probs = log_probs + torch.log(weights)

    # LogSumExp trick for numerical stability
    log_prob_actions = torch.logsumexp(weighted_log_probs, dim=-1)

    # Compute the negative log-likelihood loss
    loss = -log_prob_actions.mean()
    return lossdef mixture_gaussian_loss(policy, states, actions):
    means, log_stds, weights = policy(states)
    stds = log_stds.exp()

    # Calculate the log probabilities of the actions under each Gaussian component
    m = torch.distributions.Normal(means, stds)
    log_probs = m.log_prob(actions.unsqueeze(1).expand_as(means))

    # Sum log probabilities over action dimensions
    log_probs = log_probs.sum(-1)

    # Weight the log probabilities by the mixture weights
    weighted_log_probs = log_probs + torch.log(weights)

    # LogSumExp trick for numerical stability
    log_prob_actions = torch.logsumexp(weighted_log_probs, dim=-1)

    # Compute the negative log-likelihood loss
    loss = -log_prob_actions.mean()
    return loss# offlineRL