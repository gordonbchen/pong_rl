import random
import json

from collections import deque, namedtuple
from dataclasses import dataclass, asdict
from itertools import count
from pathlib import Path
from argparse import ArgumentParser

import gymnasium as gym
from line_profiler import profile

import matplotlib.animation as anim
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class HyperParams:
    train_episodes: int = 500
    batch_size: int = 128

    # Number of prev states to use as input (including current state).
    # Useful for determining features like velocity.
    n_state_history: int = 1

    # Learning rates.
    lr: float = 1e-4
    target_net_lr: float = 5e-3

    # Reward discount rate.
    gamma: float = 0.99

    # Epsilon-greedy scheduler.
    max_epsilon: float = 0.9
    min_epsilon: float = 0.05
    epsilon_decay: float = 1e-3

    replay_memory_maxlen: int = 10_000

    output_subdir: str = ""
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.cli_override()

        self.output_dir = Path("outputs") / self.output_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save as json.
        with open(self.output_dir / "hyper_params.json", mode="w") as f:
            json.dump(asdict(self), f, indent=4)

    def cli_override(self) -> None:
        """Override hyperparams from CLI args."""
        parser = ArgumentParser()

        parser.add_argument("--train_episodes", type=int, required=False)
        parser.add_argument("--batch_size", type=int, required=False)

        parser.add_argument("--n_state_history", type=int, required=False)

        parser.add_argument("--lr", type=float, required=False)
        parser.add_argument("--target_net_lr", type=float, required=False)

        parser.add_argument("--gamma", type=float, required=False)

        parser.add_argument("--max_epsilon", type=float, required=False)
        parser.add_argument("--min_epsilon", type=float, required=False)
        parser.add_argument("--epsilon_decay", type=float, required=False)

        parser.add_argument("--replay_memory_maxlen", type=int, required=False)
        parser.add_argument("--output_subdir", type=str, required=False)
        parser.add_argument("--device", type=str, required=False)

        args = parser.parse_args()

        for k, v in vars(args).items():
            if v is not None:
                setattr(self, k, v)


def get_epsilon(hyper_params: HyperParams, steps: int) -> float:
    """Get the epsilon value based on the number of steps taken."""
    decay = np.exp(-1.0 * steps * hyper_params.epsilon_decay)
    return hyper_params.min_epsilon + (hyper_params.max_epsilon - hyper_params.min_epsilon) * decay


def get_action(policy_net: nn.Module, state: torch.Tensor, env: gym.Env, epsilon: float) -> int:
    """Choose a random action using the greedy-epsilon policy."""
    if random.random() < epsilon:
        action = int(env.action_space.sample())
    else:
        with torch.no_grad():
            action = policy_net(state).argmax().item()

    return action


@profile
def train_step(
    policy_net: nn.Module,
    target_net: nn.Module,
    loss_func: nn.Module,
    optimizer: Optimizer,
    replay_memory: deque,
    hyper_params: HyperParams,
) -> None:
    """Perform a training step."""
    transitions = random.sample(replay_memory, hyper_params.batch_size)
    states, actions, rewards, next_states = zip(*transitions)
    actions = torch.tensor(actions, dtype=torch.int64, device=hyper_params.device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=hyper_params.device)
    states = torch.concat(states)

    # Calculate target values.
    target_values = rewards.clone()
    non_terminated_mask = [next_state is not None for next_state in next_states]
    non_terminated_next_states = torch.concat([i for i in next_states if i is not None])
    with torch.no_grad():
        next_state_values = target_net(non_terminated_next_states).max(axis=1).values
    target_values[non_terminated_mask] += hyper_params.gamma * next_state_values

    # Calculate preds of taken actions (no loss on other actions b/c no reward).
    pred_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    loss = loss_func(pred_values, target_values)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Soft update target net.
    with torch.no_grad():
        for policy_net_param, target_net_param in zip(
            policy_net.parameters(), target_net.parameters()
        ):
            updated_param = target_net_param.data + hyper_params.target_net_lr * (
                policy_net_param.data - target_net_param.data
            )
            target_net_param.data.copy_(updated_param)


def stack_batch_preprocess(
    state_history: deque[torch.Tensor], policy_net: nn.Module
) -> torch.Tensor:
    """Stack, batch, and preprocess the state history into a single batched input."""
    return policy_net.preprocess(torch.stack(list(state_history)).unsqueeze(0))


@profile
def train(
    policy_net: nn.Module,
    target_net: nn.Module,
    env: gym.Env,
    loss_func: nn.Module,
    optimizer: Optimizer,
    hyper_params: HyperParams,
) -> None:
    """Train the dqn."""
    writer = SummaryWriter(log_dir=hyper_params.output_dir, max_queue=5)

    # Move models to device.
    policy_net.to(hyper_params.device).train()
    target_net.to(hyper_params.device).train()

    # Create replay memory and transition type.
    # TODO: Prioritized experience replay.
    replay_memory = deque([], hyper_params.replay_memory_maxlen)
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])

    total_steps = 0
    for episode in range(hyper_params.train_episodes):
        episode_reward = 0
        obs, info = env.reset()

        obs = torch.tensor(obs, dtype=torch.float32, device=hyper_params.device)
        state_history = deque(
            [torch.zeros_like(obs) for i in range(hyper_params.n_state_history - 1)] + [obs],
            maxlen=hyper_params.n_state_history,
        )
        state = stack_batch_preprocess(state_history, policy_net)

        for step in count():
            # Sample action.
            epsilon = get_epsilon(hyper_params, total_steps)
            action = get_action(policy_net, state, env, epsilon)
            total_steps += 1

            # Take action, and store transition.
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated:
                # Set next state to None. This is so that the target value will
                # be only reward, and not reward + next state value.
                next_state = None
            else:
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=hyper_params.device)
                state_history.append(next_obs)
                next_state = stack_batch_preprocess(state_history, policy_net)

            transition = Transition(state, action, reward, next_state)
            replay_memory.append(transition)

            state = next_state

            # Perform a training step.
            if len(replay_memory) >= hyper_params.batch_size:
                train_step(
                    policy_net, target_net, loss_func, optimizer, replay_memory, hyper_params
                )

            if terminated or truncated:
                break

        # Log metrics.
        writer.add_scalar("episode_reward", episode_reward, episode)
        writer.add_scalar("epsilon", epsilon, episode)
        writer.add_scalar("total_steps", total_steps, episode)
        print(f"episode {episode}\treward {episode_reward}")

    writer.close()


def get_frames(policy_net: nn.Module, env: gym.Env, hyper_params: HyperParams) -> list[np.ndarray]:
    """Play a single episode and return frames."""
    obs, info = env.reset()

    obs = torch.tensor(obs, dtype=torch.float32, device=hyper_params.device)
    state_history = deque(
        [torch.zeros_like(obs) for i in range(hyper_params.n_state_history - 1)] + [obs],
        maxlen=hyper_params.n_state_history,
    )

    frames = []
    while True:
        state = stack_batch_preprocess(state_history, policy_net)
        action = get_action(policy_net, state, env, epsilon=0.0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=hyper_params.device)
        state_history.append(next_obs)

        frames.append(env.render())
        if terminated or truncated:
            break

    return frames


def show_episode(policy_net: nn.Module, env: gym.Env, hyper_params: HyperParams) -> None:
    """Show an episode of gameplay."""
    policy_net.eval()

    frames = get_frames(policy_net, env, hyper_params)

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])

    def update(frame: int):
        """Update the image."""
        img.set_data(frames[frame])
        return img

    animation = anim.FuncAnimation(fig=fig, func=update, frames=len(frames))
    animation.save(hyper_params.output_dir / "episode.mp4")
