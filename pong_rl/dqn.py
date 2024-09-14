import random

from dataclasses import dataclass
from collections import namedtuple, deque
from itertools import count
from functools import partial
from typing import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms.v2.functional import resize
from torch.optim.optimizer import Optimizer

import numpy as np
import gymnasium as gym

import matplotlib.animation as anim
import matplotlib.pyplot as plt


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

    return top_border_end, bottom_border_end


def preprocess(
    frame: np.ndarray, top_border_end: int, bottom_border_end: int
) -> torch.Tensor:
    """Crop, convert to grayscale, and size down the frame."""
    new_frame = frame[top_border_end : bottom_border_end + 1]

    new_frame = torch.tensor(new_frame, dtype=torch.float32, device=HyperParams.device)
    luminance = torch.tensor(
        [0.2989, 0.5870, 0.1140], dtype=torch.float32, device=HyperParams.device
    )
    new_frame = (new_frame @ luminance) / 255.0

    new_frame = resize(new_frame.unsqueeze(0), (32, 32))
    return new_frame


def create_dqn(n_observations: int, n_actions: int) -> nn.Module:
    """Create a dqn with given input and output shapes."""
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(n_observations, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, n_actions),
    )
    return net


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


def get_action(
    policy_net: nn.Module, state: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Choose a random action using the greedy-epsilon policy."""
    if random.random() < epsilon:
        action = torch.tensor(
            [env.action_space.sample()], dtype=torch.int64, device=HyperParams.device
        )
    else:
        with torch.no_grad():
            action = policy_net(state).argmax().unsqueeze(0)

    return action


@dataclass
class HyperParams:
    batch_size: int = 128
    lr: float = 3e-4

    gamma: float = 0.99
    target_net_lr: float = 3e-3

    min_epsilon: float = 0.05
    max_epsilon: float = 0.9
    decay: float = 0.01

    replay_memory_maxlen: int = 10_000

    train_episodes: int = 600

    device: str = "cuda"

    output_dir: Path = Path("outputs")

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)


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
    states, actions, rewards = (torch.cat(i) for i in (states, actions, rewards))

    # Calculate target values.
    target_values = torch.zeros_like(rewards) + rewards
    non_terminated_mask = [next_state is not None for next_state in next_states]
    non_terminated_next_states = torch.cat([i for i in next_states if i is not None])
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
        target_net_state_dict = target_net.state_dict()
        for k, v in policy_net.state_dict().items():
            target_net_state_dict[k] += HyperParams.target_net_lr * (
                v - target_net_state_dict[k]
            )
        target_net.load_state_dict(target_net_state_dict)


def train(
    policy_net: nn.Module,
    target_net: nn.Module,
    loss_func: nn.Module,
    optimizer: Optimizer,
    preprocessor: Callable[[np.ndarray], torch.Tensor],
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
        state = preprocessor(state)

        episode_reward = 0
        for step in count():
            # Sample an action.
            epsilon = get_epsilon(
                HyperParams.min_epsilon,
                HyperParams.max_epsilon,
                HyperParams.decay,
                episode,
            )
            action = get_action(policy_net, state, epsilon)

            # Take action and store transition.
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocessor(next_state) if not terminated else None

            episode_reward += reward
            reward = torch.tensor(
                [reward], dtype=torch.float32, device=HyperParams.device
            )
            transition = Transition(state, action, reward, next_state)
            replay_memory.append(transition)

            # Perform a training step, but fill up the buffer before training.
            if len(replay_memory) == replay_memory.maxlen:
                train_step(policy_net, target_net, loss_func, optimizer, replay_memory)

            if terminated or truncated:
                break

            state = next_state

        print(f"episode {episode}\treward {episode_reward:.5f}")


def get_frames(
    policy_net: nn.Module,
    env: gym.Env,
    preprocessor: Callable[[np.ndarray], torch.Tensor],
) -> list[np.ndarray]:
    """Play a single episode and return frames."""
    state, info = env.reset()
    state = preprocessor(state)
    frames = [state]

    while True:
        action = get_action(policy_net, state, epsilon=0.0)
        new_state, reward, terminated, truncated, info = env.step(action.item())
        new_state = preprocessor(new_state)

        frames.append(new_state)
        if terminated or truncated:
            break

        state = new_state

        env.close()
    return [f.cpu().numpy()[0] for f in frames]


def show_episode(
    policy_net: nn.Module,
    env: gym.Env,
    preprocessor: Callable[[np.ndarray], torch.Tensor],
) -> None:
    """Show an episode of gameplay."""
    policy_net.eval()

    frames = get_frames(policy_net, env, preprocessor)

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

    # Preprocess a single state to find net input shape.
    state, info = env.reset()
    top_border_end, bottom_border_end = find_border_ends(state)
    preprocessor = partial(
        preprocess, top_border_end=top_border_end, bottom_border_end=bottom_border_end
    )
    processed_frame = preprocessor(state)

    # Create policy and target nets. Copy policy weights to target net.
    n_observations = len(processed_frame.flatten())
    policy_net = create_dqn(n_observations, env.action_space.n)

    target_net = create_dqn(n_observations, env.action_space.n)
    target_net.load_state_dict(policy_net.state_dict())

    # Optimizer and loss func.
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=HyperParams.lr)
    loss_func = nn.HuberLoss()

    plot_epsilon()

    # Train.
    # BUG: reward not improving.
    train(policy_net, target_net, loss_func, optimizer, preprocessor)

    # Show an episode.
    show_episode(policy_net, env, preprocessor)
