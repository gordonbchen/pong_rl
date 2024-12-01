"""Train an RL model on cartpole using PPO."""

from pathlib import Path

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

import matplotlib.pyplot as plt
import matplotlib.animation as anim


# Hyperparams.
N_HIDDEN_LAYERS = 4
N_NEURONS = 256
LR = 3e-4

TRAIN_EPISODES = 500
EPOCHS_PER_EPISODE = 32

PROB_RATIO_CLIP_FACTOR = 0.1
ENTROPY_LOSS_COEFF = 0.1
CRITIC_LOSS_COEFF = 1.0

TIME_DISCOUNT_FACTOR_GAMMA = 0.99
GAE_DISCOUNT_FACTOR_LAMBDA = 0.95

OUTPUT_DIR = Path("outputs/ppo/cartpole/test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ActorCritic(nn.Module):
    """An actor and critic with a shared backbone."""

    def __init__(self, n_obs: int, n_actions: int, n_hidden_layers: int, n_neurons: int) -> None:
        """Initialize the actor critic."""
        super().__init__()

        in_layer = nn.Sequential(nn.Linear(n_obs, n_neurons), nn.ReLU())
        hidden_layers = nn.Sequential(
            *[nn.Linear(n_neurons, n_neurons), nn.ReLU()] * n_hidden_layers
        )
        self.backbone = nn.Sequential(in_layer, hidden_layers)

        self.actor = nn.Sequential(nn.Linear(n_neurons, n_actions), nn.Softmax(dim=-1))
        self.critic = nn.Linear(n_neurons, 1)

    def forward(self, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward the actor critic model.
        Return a probability distribution over actions and the state value.
        """
        z = self.backbone(xb)
        return self.actor(z), self.critic(z)

    def get_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over actions."""
        z = self.backbone(states)
        return self.actor(z)


@torch.no_grad()
def rollout_episode(
    actor_critic: ActorCritic, env: gym.Env
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rollout an episode in the env using the actor_critic model.
    Returns states (t, n_obs), actions (t, 1), prob of each action (t, 1), and rewards (t, 1).
    TODO: roll out fixed # steps instead of 1 episode?
    """
    state, info = env.reset()

    states = []
    actions = []
    action_probs = []
    rewards = []

    while True:
        state = torch.tensor(state, dtype=torch.float32, device="cuda")
        probs = actor_critic.get_action_probs(state.unsqueeze(0))[0]
        action = torch.multinomial(probs, 1)

        states.append(state)
        state, reward, terminated, truncated, info = env.step(action.item())

        actions.append(action)
        action_probs.append(probs[action])
        rewards.append(reward)

        if terminated or truncated:
            break

    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(action_probs),
        torch.tensor(rewards, dtype=torch.float32, device="cuda").unsqueeze(1),
    )


def optimize(
    actor_critic: ActorCritic,
    optim: Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    prev_action_probs: torch.Tensor,
    rewards: torch.Tensor,
    tensorboard_log: SummaryWriter,
    n_training_steps: int,
) -> torch.Tensor:
    """
    Perform an optimization step on the actor critic over a rolled out episode.
    Returns action_probs to calculate the probability ratio when performing another optimize step.
    TODO: optimize in mini-batches?
    """
    # Calculate new probabilities for action, probability ratio, and mean entropy.
    probs, values = actor_critic(states)
    action_probs = probs.gather(1, actions)
    prob_ratios = action_probs / prev_action_probs
    mean_entropy = (-1.0 * action_probs * action_probs.log()).sum(dim=-1).mean()

    # Calculate advantages using GAE.
    td = rewards - values
    td[:-1] += TIME_DISCOUNT_FACTOR_GAMMA * values[1:]

    advantages = td
    for i in reversed(range(len(advantages) - 1)):
        advantages[i] += GAE_DISCOUNT_FACTOR_LAMBDA * TIME_DISCOUNT_FACTOR_GAMMA * advantages[i + 1]

    # Calculate loss.
    unclipped_actor_loss = prob_ratios * advantages.detach()
    clipped_prob_ratios = torch.clip(
        prob_ratios, 1.0 - PROB_RATIO_CLIP_FACTOR, 1.0 + PROB_RATIO_CLIP_FACTOR
    )
    clipped_actor_loss = clipped_prob_ratios * advantages.detach()
    actor_loss = -1.0 * torch.min(unclipped_actor_loss, clipped_actor_loss).mean()

    critic_loss = CRITIC_LOSS_COEFF * 0.5 * (advantages**2.0).mean()
    entropy_loss = -1.0 * ENTROPY_LOSS_COEFF * mean_entropy
    loss = actor_loss + critic_loss + entropy_loss

    # Backprop and optimize.
    optim.zero_grad()
    loss.backward()
    optim.step()

    # Log.
    tensorboard_log.add_scalars(
        "loss",
        {
            "actor": actor_loss.item(),
            "critic": critic_loss.item(),
            "entropy": entropy_loss.item(),
            "total": loss.item(),
        },
        n_training_steps,
    )
    tensorboard_log.add_scalar("prob_ratio", prob_ratios.mean().item(), n_training_steps)
    tensorboard_log.add_scalar("entropy", mean_entropy.item(), n_training_steps)

    return action_probs.detach()


@torch.no_grad()
def show_episode(actor_critic: ActorCritic, env: gym.Env) -> None:
    """Roll out an episode in eval mode and save an animation."""
    # Roll out episode.
    actor_critic.eval()

    state, info = env.reset()
    frames = [env.render()]

    while True:
        state = torch.tensor(state, dtype=torch.float32, device="cuda")
        probs = actor_critic.get_action_probs(state.unsqueeze(0))[0]
        action = torch.multinomial(probs, num_samples=1).item()
        state, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())

        if terminated or truncated:
            break

    # Plot and save animation.
    fig, ax = plt.subplots()
    img = ax.imshow(frames[0])

    def update(frame: int):
        """Update the image."""
        img.set_data(frames[frame])
        return img

    animation = anim.FuncAnimation(fig=fig, func=update, frames=len(frames))
    animation.save(OUTPUT_DIR / "episode.mp4")


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor_critic = ActorCritic(n_obs, n_actions, N_HIDDEN_LAYERS, N_NEURONS).to("cuda").train()
    optim = Adam(actor_critic.parameters(), lr=LR)

    tensorboard_log = SummaryWriter(OUTPUT_DIR)

    max_episode_reward = float("-inf")
    n_training_steps = 0

    for episode in range(TRAIN_EPISODES):
        states, actions, action_probs, rewards = rollout_episode(actor_critic, env)

        episode_reward = rewards.sum().item()
        print(f"Episode={episode}, steps={n_training_steps}, reward={episode_reward}")
        tensorboard_log.add_scalar("reward", episode_reward, n_training_steps)

        if episode_reward > max_episode_reward:
            max_episode_reward = episode_reward
            show_episode(actor_critic, env)
            actor_critic.train()

        for epoch in range(EPOCHS_PER_EPISODE):
            n_training_steps += 1
            action_probs = optimize(
                actor_critic,
                optim,
                states,
                actions,
                action_probs,
                rewards,
                tensorboard_log,
                n_training_steps,
            )

    tensorboard_log.close()
