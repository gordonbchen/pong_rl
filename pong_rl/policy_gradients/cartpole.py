"""
Pseudocode:
1. Define policy network. Observation space -> n_actions. Outputs probability distribution.
2. Play episode with policy. Collect states, actions, rewards.
Need gradient of log probabilities wrt policy network params and true return after action.
3. Calculate gradient of objective function (expected return).
= gradient of log probability of taking action * return after action.
Since pytorch like gradient descent, turn objective function into loss function.
4. Update policy network params.
"""

import torch
import torch.nn as nn
from torch.optim import Adam

import gymnasium as gym

from policy_gradients import train, show_episode, HyperParams


class PolicyNet(nn.Module):
    """A policy network."""

    def __init__(self, n_obs: int, n_actions: int, n_hidden_layers: int, n_neurons: int) -> None:
        """Initialilze the policy network."""
        super().__init__()

        in_linear = nn.Sequential(nn.Linear(n_obs, n_neurons, bias=True), nn.ReLU())
        hidden_layers = nn.Sequential(
            *[nn.Linear(n_neurons, n_neurons, bias=True), nn.ReLU()] * (n_hidden_layers - 1)
        )
        out_linear = nn.Linear(n_neurons, n_actions, bias=True)

        self.net = nn.Sequential(in_linear, hidden_layers, out_linear)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Forward the policy net. Returns probabilities."""
        logits = self.net(xb)
        return torch.softmax(logits, dim=1)


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, info = env.reset()

    hyper_params = HyperParams()

    policy_net = PolicyNet(
        n_obs=len(obs), n_actions=env.action_space.n, n_hidden_layers=3, n_neurons=256
    )
    optim = Adam(policy_net.parameters(), lr=hyper_params.lr)

    train(policy_net, env, optim, hyper_params)
    show_episode(policy_net, env, hyper_params)
