import random

from dataclasses import dataclass
from collections import namedtuple, deque
from itertools import count
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms.v2.functional import resize
from torch.optim.optimizer import Optimizer

import numpy as np
import gymnasium as gym

import matplotlib.animation as anim
import matplotlib.pyplot as plt

from line_profiler import profile


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


@dataclass
class HyperParams:
    batch_size: int = 64
    lr: float = 3e-4

    gamma: float = 0.99
    target_net_lr: float = 3e-3

    min_epsilon: float = 0.05
    max_epsilon: float = 0.9
    decay: float = 0.01

    replay_memory_maxlen: int = 10_000

    train_episodes: int = 500

    device: str = "cuda"

    output_dir: Path = Path("outputs")

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)


LUMINANCE = torch.tensor(
    [0.2989, 0.5870, 0.1140], dtype=torch.float32, device=HyperParams.device
)


@profile
def preprocess(
    batch_frames: torch.Tensor, top_border_end: int, bottom_border_end: int
) -> torch.Tensor:
    """Crop, convert to grayscale, and downsize."""
    z = batch_frames[:, top_border_end : bottom_border_end + 1] / 255.0

    z = z @ LUMINANCE

    z = resize(z.unsqueeze(1), (32, 32))
    return z


class DQN(nn.Module):
    """Deep Q Network."""

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
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Forward the model."""
        z = preprocess(xb, self.top_border_end, self.bottom_border_end)

        z = self.conv1(z)
        z = self.conv2(z)
        z = self.linear(z)
        return z


def get_epsilon(
    min_epsilon: float, max_epsilon: float, decay: float, episode: int
) -> float:
    """Get the epsilon value based on the current episode."""
    return min_epsilon + (max_epsilon - min_epsilon) * (np.exp(-1.0 * episode * decay))


def plot_epsilon() -> None:
    """Plot epsilon values over training episodes."""
    episodes = np.arange(HyperParams.train_episodes)
    epsilons = get_epsilon(
        HyperParams.min_epsilon, HyperParams.max_epsilon, HyperParams.decay, episodes
    )

    plt.style.use("bmh")

    plt.plot(episodes, epsilons)
    plt.hlines(
        HyperParams.min_epsilon,
        xmin=0,
        xmax=HyperParams.train_episodes,
        linestyles="dashed",
    )

    plt.xlabel("episode")
    plt.ylabel("epsilon")

    plt.savefig(HyperParams.output_dir / "epsilon.png")


def get_action(policy_net: nn.Module, state: torch.Tensor, epsilon: float) -> int:
    """Choose a random action using the greedy-epsilon policy."""
    if random.random() < epsilon:
        action = int(env.action_space.sample())
    else:
        with torch.no_grad():
            action = policy_net(state.unsqueeze(0)).argmax().item()

    return action


@profile
def train_step(
    policy_net: nn.Module,
    target_net: nn.Module,
    loss_func: nn.Module,
    optimizer: Optimizer,
    replay_memory: deque,
) -> None:
    """Perform a training step."""
    transitions = random.sample(replay_memory, HyperParams.batch_size)
    states, actions, rewards, next_states = zip(*transitions)
    actions = torch.tensor(actions, dtype=torch.int64, device=HyperParams.device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=HyperParams.device)
    states = torch.stack(states)

    # Calculate target values.
    target_values = torch.zeros_like(rewards) + rewards
    non_terminated_mask = [next_state is not None for next_state in next_states]
    non_terminated_next_states = torch.stack([i for i in next_states if i is not None])
    with torch.no_grad():
        next_state_values = target_net(non_terminated_next_states).max(axis=1)[0]
    target_values[non_terminated_mask] += HyperParams.gamma * next_state_values

    # Calculate preds of taken actions (no loss on other actions b/c no reward).
    optimizer.zero_grad()
    pred_values = policy_net(states).gather(1, actions.unsqueeze(0))

    loss = loss_func(pred_values, target_values.unsqueeze(0))
    loss.backward()
    optimizer.step()

    # Soft update target net.
    with torch.no_grad():
        for policy_net_param, target_net_param in zip(
            policy_net.parameters(), target_net.parameters()
        ):
            target_net_param += HyperParams.target_net_lr * (
                policy_net_param - target_net_param
            )


@profile
def train(
    policy_net: nn.Module,
    target_net: nn.Module,
    loss_func: nn.Module,
    optimizer: Optimizer,
) -> None:
    """Train the dqn."""
    # Move models to device.
    policy_net.to(HyperParams.device).train()
    target_net.to(HyperParams.device).train()

    # Create replay memory and transition type.
    replay_memory = deque([], HyperParams.replay_memory_maxlen)
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])

    for episode in range(HyperParams.train_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=HyperParams.device)
        prev_curr_state_diff = torch.zeros(
            state.shape, dtype=torch.float32, device=HyperParams.device
        )

        episode_reward = 0
        for step in count():
            # Sample an action.
            epsilon = get_epsilon(
                HyperParams.min_epsilon,
                HyperParams.max_epsilon,
                HyperParams.decay,
                episode,
            )
            action = get_action(policy_net, prev_curr_state_diff, epsilon)

            # Take action and store transition.
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated:
                curr_next_state_diff = None
            else:
                next_state = torch.tensor(
                    next_state, dtype=torch.float32, device=HyperParams.device
                )
                curr_next_state_diff = next_state - state

            # TODO: state stack vs diff?
            transition = Transition(
                prev_curr_state_diff, action, reward, curr_next_state_diff
            )
            replay_memory.append(transition)

            # Perform a training step, but fill up the buffer before training.
            if len(replay_memory) == replay_memory.maxlen:
                train_step(policy_net, target_net, loss_func, optimizer, replay_memory)

            if terminated or truncated:
                break

            prev_curr_state_diff = curr_next_state_diff
            state = next_state

        print(f"episode {episode}\treward {episode_reward:.5f}")


def get_frames(
    policy_net: nn.Module, env: gym.Env, top_border_end: int, bottom_border_end: int
) -> list[np.ndarray]:
    """Play a single episode and return frames."""
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=HyperParams.device)
    prev_state = torch.zeros_like(state)
    frames = []

    while True:
        action = get_action(policy_net, state - prev_state, epsilon=0.0)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device=HyperParams.device
        )

        frame = preprocess(
            (next_state - state).unsqueeze(0), top_border_end, bottom_border_end
        )[0][0]
        frames.append(frame)
        if terminated or truncated:
            break

        prev_state = state
        state = next_state

    return [f.cpu().numpy() for f in frames]


def show_episode(
    policy_net: nn.Module, env: gym.Env, top_border_end: int, bottom_border_end: int
) -> None:
    """Show an episode of gameplay."""
    policy_net.eval()

    frames = get_frames(policy_net, env, top_border_end, bottom_border_end)

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])

    def update(frame: int):
        """Update the image."""
        img.set_data(frames[frame])
        return img

    animation = anim.FuncAnimation(fig=fig, func=update, frames=len(frames))
    animation.save(HyperParams.output_dir / "dqn.mp4")


if __name__ == "__main__":
    # Create env.
    env = gym.make(
        "ALE/Pong-v5", mode=0, difficulty=0, obs_type="rgb", render_mode="rgb_array"
    )

    # Create env and find cropping inds.
    state, info = env.reset()
    top_border_end, bottom_border_end = find_border_ends(state)

    # Create policy and target nets. Copy policy weights to target net.
    policy_net = DQN(env.action_space.n, top_border_end, bottom_border_end)
    target_net = DQN(env.action_space.n, top_border_end, bottom_border_end)
    target_net.load_state_dict(policy_net.state_dict())

    # Optimizer and loss func.
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=HyperParams.lr)
    loss_func = nn.HuberLoss()

    plot_epsilon()

    # Train.
    # BUG: reward not improving.
    train(policy_net, target_net, loss_func, optimizer)

    # Show an episode.
    show_episode(policy_net, env, top_border_end, bottom_border_end)
