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
        self,
        n_input_channels: int,
        n_actions: int,
        top_border_end: int,
        bottom_border_end: int,
    ) -> None:
        """Initialize the network."""
        super().__init__()
        self.top_border_end = top_border_end
        self.bottom_border_end = bottom_border_end

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                n_input_channels,
                32,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
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
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def preprocess(self, xb: torch.Tensor) -> torch.Tensor:
        """Preprocess the inputs. Crop, grayscale, and size down."""
        z = xb[:, :, self.top_border_end : self.bottom_border_end + 1] / 255.0
        return resize(z, (32, 32))

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Forward the model. Inputs should have already been preprocessed."""
        z = self.conv1(xb)
        z = self.conv2(z)
        return self.linear(z)


if __name__ == "__main__":
    # Create env.
    env = gym.make(
        "ALE/Pong-v5",
        mode=0,
        difficulty=0,
        obs_type="grayscale",
        render_mode="rgb_array",
    )
    obs, info = env.reset()

    # Create policy and target nets. Copy policy weights to target net.
    n_actions = env.action_space.n
    top_border_end, bottom_border_end = find_border_ends(obs)

    hyper_params = HyperParams(
        train_episodes=500,
        batch_size=32,
        n_state_history=4,
        lr=3e-4,
        target_net_lr=5e-3,
        gamma=0.99,
        max_epsilon=1.0,
        min_epsilon=0.01,
        epsilon_decay=1.5e-5,
        replay_memory_maxlen=int(1e4),
        output_subdir="pong_dqn/cli",
        device="cuda",
        use_cli_args=True,
    )

    policy_net = ConvDQN(hyper_params.n_state_history, n_actions, top_border_end, bottom_border_end)
    target_net = ConvDQN(hyper_params.n_state_history, n_actions, top_border_end, bottom_border_end)
    target_net.load_state_dict(policy_net.state_dict())

    # Optimizer and loss func.
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=hyper_params.lr, amsgrad=True)
    loss_func = nn.HuberLoss()

    # Train and show an episode.
    train(policy_net, target_net, env, loss_func, optimizer, hyper_params)
    show_episode(policy_net, env, hyper_params)

    # Save model.
    torch.save(
        {
            "policy_net_state_dict": policy_net.state_dict(),
            "target_net_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        hyper_params.output_dir / "saved_models.tar",
    )
