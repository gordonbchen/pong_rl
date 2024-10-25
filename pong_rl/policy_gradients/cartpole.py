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

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym


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

    def act(self, obs: np.ndarray) -> tuple[int, torch.Tensor]:
        """
        Forward the model to get a probability distribution over actions.
        Sample random action from distribution.
        Return action and log of probability.
        """
        obs = torch.tensor(obs).to("cuda", dtype=torch.float32).unsqueeze(0)
        probs = policy_net(obs).squeeze(0)

        action = torch.multinomial(probs, num_samples=1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob


def train(episodes: int, policy_net: PolicyNet, env: gym.Env, optim: Optimizer):
    """Train the policy network."""
    summary_writer = SummaryWriter(log_dir="outputs/policy_gradients/cartpole", max_queue=5)

    policy_net.train().to("cuda")

    for episode in range(episodes):
        # Collect log_probs and rewards.
        obs, _ = env.reset()
        log_probs, rewards = [], []

        while True:
            action, log_prob = policy_net.act(obs)
            log_probs.append(log_prob)

            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Calculate and normalize returns.
        # NOTE: returns are reversed.
        returns = torch.empty(len(rewards), dtype=torch.float32, device="cuda")
        for i in range(len(rewards)):
            future_return = 0.0 if (i == 0) else returns[i - 1]
            returns[i] = rewards[-1 - i] + (0.99 * future_return)

        returns = (returns - returns.mean()) / (returns.std() + torch.finfo(torch.float32).eps)

        # Calculate gradient of returns and optimize.
        log_probs = torch.stack(tuple(reversed(log_probs)))
        loss = (-1.0 * log_probs * returns).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Logging.
        total_rewards = sum(rewards)
        summary_writer.add_scalar("rewards", total_rewards, episode)
        summary_writer.add_scalar("loss", loss, episode)

        print(f"episode {episode}: rewards = {total_rewards}")

    summary_writer.close()


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, info = env.reset()

    policy_net = PolicyNet(
        n_obs=len(obs), n_actions=env.action_space.n, n_hidden_layers=3, n_neurons=256
    )
    optim = Adam(policy_net.parameters(), lr=3e-4)

    train(1_000, policy_net, env, optim)
