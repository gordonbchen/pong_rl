from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym


@dataclass
class HyperParams:
    train_episodes: int = 1_000
    lr: float = 3e-4
    gamma: float = 0.99
    output_dir: str = "outputs/policy_gradients/test"

    def __post_init__(self) -> None:
        """Override params using cli args and create output dir and save hyperparams."""
        self.cli_override()
        self.create_output_dir_and_save()

    def cli_override(self) -> None:
        """Override hyperparams from CLI args."""
        parser = ArgumentParser()
        for k, v in asdict(self).items():
            parser.add_argument(f"--{k}", type=type(v), default=v)
        args = parser.parse_args()

        for k, v in vars(args).items():
            setattr(self, k, v)

    def create_output_dir_and_save(self) -> None:
        """Create the output dir and save hyperparams."""
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "hyper_params.json", mode="w") as f:
            json.dump(asdict(self), f, indent=4)

        self.output_dir = output_dir  # HACK: set after json dump b/c Path is not serializable.


def act(policy_net: nn.Module, obs: np.ndarray) -> tuple[int, torch.Tensor]:
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


def train(policy_net: nn.Module, env: gym.Env, optim: Optimizer, hyper_params: HyperParams):
    """Train the policy network."""
    summary_writer = SummaryWriter(log_dir=hyper_params.output_dir, max_queue=5)

    policy_net.train().to("cuda")

    for episode in range(hyper_params.train_episodes):
        # Collect log_probs and rewards.
        obs, _ = env.reset()
        log_probs, rewards = [], []

        while True:
            action, log_prob = act(policy_net, obs)
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
            returns[i] = rewards[-1 - i] + (hyper_params.gamma * future_return)

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


@torch.no_grad()
def get_frames(policy_net: nn.Module, env: gym.Env) -> list[np.ndarray]:
    """Play a single episode and return frames."""
    frames = []

    obs, _ = env.reset()
    while True:
        action, _ = act(policy_net, obs)
        obs, _, terminated, truncated, _ = env.step(action)

        frames.append(env.render())
        if terminated or truncated:
            break

    return frames


def show_episode(policy_net: nn.Module, env: gym.Env, hyper_params: HyperParams) -> None:
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
    animation.save(hyper_params.output_dir / "episode.mp4")
