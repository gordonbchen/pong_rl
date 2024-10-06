import torch
import torch.nn as nn
from torchvision.transforms.v2.functional import resize

import numpy as np
import gymnasium as gym

from train_funcs import HyperParams, show_episode, train


def find_border_ends(state: np.ndarray) -> tuple[int, int]:
    """
    Find the inds where the top and bottom borders end (for cropping).
    Returns the first non-border row ind.
    """
    border_color = state[-1, -1]

    in_border = False
    for i, color in enumerate(state[:, 0]):
        if (color == border_color).all():
            in_border = True
        elif in_border:
            top_border_end = i
            break
    else:
        raise AssertionError("No top border end found.")

    for i, color in enumerate(state[::-1, 0]):
        if (color != border_color).any():
            bottom_border_end = len(state) - 1 - i
            break
    else:
        raise AssertionError("No bottom border end found")

    # TODO: crop sides beyond paddles.
    return top_border_end, bottom_border_end


class ConvDQN(nn.Module):
    """Convolutional Deep Q Network."""

    def __init__(
        self, n_actions: int, top_border_end: int, bottom_border_end: int
    ) -> None:
        """Initialize the network."""
        super().__init__()
        self.top_border_end = top_border_end
        self.bottom_border_end = bottom_border_end

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Forward the model."""
        # Preprocess.
        z = xb[:, self.top_border_end : self.bottom_border_end + 1] / 255.0
        z = resize(z.unsqueeze(1), (32, 32))

        z = self.conv1(z)
        z = self.conv2(z)
        z = self.linear(z)
        return z


if __name__ == "__main__":
    # Create env.
    env = gym.make(
        "ALE/Pong-v5",
        mode=0,
        difficulty=0,
        obs_type="grayscale",
        render_mode="rgb_array",
    )
    state, info = env.reset()

    # Create policy and target nets. Copy policy weights to target net.
    n_actions = env.action_space.n
    top_border_end, bottom_border_end = find_border_ends(state)

    policy_net = ConvDQN(n_actions, top_border_end, bottom_border_end)
    target_net = ConvDQN(n_actions, top_border_end, bottom_border_end)
    target_net.load_state_dict(policy_net.state_dict())

    hyper_params = HyperParams(
        train_episodes=500,
        batch_size=128,
        use_state_diff=True,
        lr=1e-4,
        target_net_lr=5e-3,
        gamma=0.99,
        max_epsilon=0.9,
        min_epsilon=0.05,
        epsilon_decay=1e-5,
        replay_memory_maxlen=50_000,
        output_subdir="pong_dqn_state_diff",  # TODO: Make this a CLI arg.
        device="cuda",
    )

    # Optimizer and loss func.
    optimizer = torch.optim.AdamW(
        policy_net.parameters(), lr=hyper_params.lr, amsgrad=True
    )
    loss_func = nn.HuberLoss()

    # Train and show an episode.
    train(policy_net, target_net, env, loss_func, optimizer, hyper_params)
    show_episode(policy_net, env, hyper_params)
