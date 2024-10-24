import torch
import torch.nn as nn

import gymnasium as gym

from train_funcs import HyperParams, show_episode, train


class MLPDQN(nn.Module):
    """MLP Deep Q Network."""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        """Initialize the network."""
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def preprocess(self, xb: torch.Tensor) -> torch.Tensor:
        """Preprocess the inputs."""
        # Squeeze to support passing history.
        # Stacking during state creation and unsqueezing
        # during action forward gives duplicate batch dim.
        return xb.squeeze(1)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Forward the model. Inputs should already be preprocessed."""
        return self.linear(xb)


if __name__ == "__main__":
    # Create env.
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, info = env.reset()

    # Create policy and target nets. Copy policy weights to target net.
    n_observations = len(obs)
    n_actions = env.action_space.n

    policy_net = MLPDQN(n_observations, n_actions)
    target_net = MLPDQN(n_observations, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    # Optimizer and loss func.
    hyper_params = HyperParams(
        train_episodes=500,
        batch_size=128,
        n_state_history=1,
        lr=1e-4,
        target_net_lr=5e-3,
        gamma=0.99,
        max_epsilon=0.9,
        min_epsilon=0.05,
        epsilon_decay=1e-3,
        replay_memory_maxlen=10_000,
        output_subdir="cartpole_dqn/speed",
        device="cuda",
        use_cli_args=True,
    )

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=hyper_params.lr, amsgrad=True)
    loss_func = nn.HuberLoss()

    # Train and show an episode.
    train(policy_net, target_net, env, loss_func, optimizer, hyper_params)
    show_episode(policy_net, env, hyper_params)
