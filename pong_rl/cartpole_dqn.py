import random

from dataclasses import dataclass
from collections import namedtuple, deque
from itertools import count
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

import numpy as np
import gymnasium as gym

import matplotlib.animation as anim
import matplotlib.pyplot as plt

from line_profiler import profile


@dataclass
class HyperParams:
    batch_size = 128
    lr = 1e-4
    gradient_clip_value = 2.5

    gamma = 0.99
    target_net_lr = 5e-3

    max_epsilon = 0.9
    min_epsilon = 0.05
    decay = 1e-3

    replay_memory_maxlen = 10_000

    train_episodes = 500
    device = "cuda"

    output_dir = Path("outputs") / "cartpole_dqn"
    output_dir.mkdir(parents=True, exist_ok=True)


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


def get_epsilon(
    min_epsilon: float, max_epsilon: float, decay: float, steps: int
) -> float:
    """Get the epsilon value based on the number of steps taken."""
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-1.0 * steps * decay)


def get_action(
    policy_net: nn.Module, state: torch.Tensor, env: gym.Env, epsilon: float
) -> int:
    """Choose a random action using the greedy-epsilon policy."""
    if random.random() < epsilon:
        action = int(env.action_space.sample())
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=HyperParams.device)
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
    states = torch.tensor(
        np.stack(states), dtype=torch.float32, device=HyperParams.device
    )

    # Calculate target values.
    target_values = rewards.clone()
    non_terminated_mask = [next_state is not None for next_state in next_states]
    non_terminated_next_states = torch.tensor(
        np.stack([i for i in next_states if i is not None]),
        dtype=torch.float32,
        device=HyperParams.device,
    )
    with torch.no_grad():
        next_state_values = target_net(non_terminated_next_states).max(axis=1).values
    target_values[non_terminated_mask] += HyperParams.gamma * next_state_values

    # Calculate preds of taken actions (no loss on other actions b/c no reward).
    optimizer.zero_grad()
    pred_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

    loss = loss_func(pred_values, target_values)
    loss.backward()

    # Clip gradient.
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(), HyperParams.gradient_clip_value
    )
    optimizer.step()

    # Soft update target net.
    with torch.no_grad():
        for policy_net_param, target_net_param in zip(
            policy_net.parameters(), target_net.parameters()
        ):
            updated_param = target_net_param.data + HyperParams.target_net_lr * (
                policy_net_param.data - target_net_param.data
            )
            target_net_param.data.copy_(updated_param)


@profile
def train(
    policy_net: nn.Module,
    target_net: nn.Module,
    env: gym.Env,
    loss_func: nn.Module,
    optimizer: Optimizer,
) -> np.ndarray:
    """Train the dqn."""
    # Move models to device.
    policy_net.to(HyperParams.device).train()
    target_net.to(HyperParams.device).train()

    # Create replay memory and transition type.
    replay_memory = deque([], HyperParams.replay_memory_maxlen)
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])

    # Track episode rewards.
    rewards = np.empty(HyperParams.train_episodes, dtype=np.float32)
    total_steps = 0

    for episode in range(HyperParams.train_episodes):
        episode_reward = 0
        state, info = env.reset()

        for step in count():
            # Sample action.
            epsilon = get_epsilon(
                HyperParams.min_epsilon,
                HyperParams.max_epsilon,
                HyperParams.decay,
                total_steps,
            )
            action = get_action(policy_net, state, env, epsilon)
            total_steps += 1

            # Take action, and store transition.
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated:
                # Set next state to None. This is so that the target value will
                # be only reward, and not reward + next state value.
                next_state = None

            transition = Transition(state, action, reward, next_state)
            replay_memory.append(transition)
            state = next_state

            # Perform a training step.
            if len(replay_memory) >= HyperParams.batch_size:
                train_step(policy_net, target_net, loss_func, optimizer, replay_memory)

            if terminated or truncated:
                break

        # Display episode and reward.
        rewards[episode] = episode_reward
        print(f"episode {episode}\treward {episode_reward:.5f}")

    return rewards


def plot_rewards(rewards: np.ndarray) -> None:
    """Plot training rewards over training episodes."""
    plt.plot(rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")

    plt.savefig(HyperParams.output_dir / "rewards.png")


def get_frames(policy_net: nn.Module, env: gym.Env) -> list[np.ndarray]:
    """Play a single episode and return frames."""
    state, info = env.reset()
    frames = []

    while True:
        action = get_action(policy_net, state, env, epsilon=0.0)
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state

        frames.append(env.render())
        if terminated or truncated:
            break

    return frames


def show_episode(policy_net: nn.Module, env: gym.Env) -> None:
    """Show an episode of gameplay."""
    policy_net.eval()

    frames = get_frames(policy_net, env)

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])

    def update(frame: int):
        """Update the image."""
        img.set_data(frames[frame])
        return img

    animation = anim.FuncAnimation(fig=fig, func=update, frames=len(frames))
    animation.save(HyperParams.output_dir / "episode.mp4")


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
    optimizer = torch.optim.AdamW(
        policy_net.parameters(), lr=HyperParams.lr, amsgrad=True
    )
    loss_func = nn.HuberLoss()

    # Train.
    rewards = train(policy_net, target_net, env, loss_func, optimizer)
    plot_rewards(rewards)

    # Show an episode.
    show_episode(policy_net, env)
