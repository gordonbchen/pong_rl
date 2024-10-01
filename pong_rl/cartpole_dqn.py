import torch
import torch.nn as nn

import gymnasium as gym

from train_funcs import HyperParams, plot_rewards, show_episode, train


class DQN(nn.Module):
    """Deep Q Network."""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        """Initialize the network."""
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Forward the model."""
        z = self.linear(xb)
        return z


if __name__ == "__main__":
    # Create env.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state, info = env.reset()

    # Create policy and target nets. Copy policy weights to target net.
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions)
    target_net = DQN(n_observations, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    # Optimizer and loss func.
    hyper_params = HyperParams(
        train_episodes=500,
        batch_size=128,
        lr=1e-4,
        target_net_lr=5e-3,
        gamma=0.99,
        max_epsilon=0.9,
        min_epsilon=0.05,
        epsilon_decay=1e-3,
        replay_memory_maxlen=10_000,
        gradient_clip_value=100.0,
        output_subdir="cartpole_dqn",
        device="cuda",
    )

    optimizer = torch.optim.AdamW(
        policy_net.parameters(), lr=hyper_params.lr, amsgrad=True
    )
    loss_func = nn.HuberLoss()

    # Train.
    rewards = train(policy_net, target_net, env, loss_func, optimizer, hyper_params)
    plot_rewards(rewards, hyper_params.output_dir)

    # Show an episode.
    show_episode(policy_net, env, hyper_params)
