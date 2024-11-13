"""Train an Advantage Actor Critic (A2C) model on cartpole."""

from pathlib import Path

import gymnasium as gym

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from torch.utils.tensorboard import SummaryWriter


LR = 1e-4
GAMMA = 0.99
TRAIN_EPISODES = 2_000
N_HIDDEN_LAYERS = 3
N_NEURONS = 256
OUTPUT_DIR = Path("outputs/a2c/cartpole")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class ActorCritic(nn.Module):
    """An actor-critic model with a shared backbone."""

    def __init__(self, n_obs: int, n_actions: int, n_hidden_layers: int, n_neurons: int) -> None:
        """Init actor model."""
        super().__init__()

        # Shared backbone for actor and critic.
        in_layer = nn.Sequential(nn.Linear(n_obs, n_neurons), nn.ReLU())
        hidden_layers = nn.Sequential(
            *[nn.Linear(n_neurons, n_neurons), nn.ReLU()] * (n_hidden_layers)
        )
        self.backbone = nn.Sequential(in_layer, hidden_layers)

        self.actor = nn.Sequential(nn.Linear(n_neurons, n_actions), nn.Softmax(dim=-1))
        self.critic = nn.Linear(n_neurons, 1)

    def forward(self, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns probability distribution over actions, and value of state."""
        z = self.backbone(xb)
        return self.actor(z), self.critic(z)


def train(actor_critic: ActorCritic, optim: Optimizer, env: gym.Env, train_episodes: int) -> None:
    """
    Train using A2C.

    For timestep t and state s:
    * Get action log probs from policy model: probs = policy_model(s)
    * Sample action from log probs: a = multinomial(probs)
    * Take action, get reward and next state: s --a--> r, s'
    * Calculate td target and advantage using value model
        * td_target = r + (gamma * V(s'))
        * advantage = td_target - V(s)
    * Optimize policy: loss = -log(probs) * advantage
    * Update value model: loss = 0.5 * (advantage ** 2)
    """
    actor_critic.train().to("cuda")
    summary_writer = SummaryWriter(OUTPUT_DIR)

    for episode in range(train_episodes):
        episode_reward = 0

        obs, _ = env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device="cuda").unsqueeze(0)
        while True:
            action_probs, curr_value = actor_critic(state)
            action_probs = action_probs.squeeze(0)
            action = torch.multinomial(action_probs, 1).item()

            # Take action.
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            next_state = torch.tensor(next_obs, dtype=torch.float32, device="cuda").unsqueeze(0)

            # Calculate advantage.
            with torch.no_grad():
                _, next_value = actor_critic(next_state)
            td_target = reward + (GAMMA * next_value) * (1 - terminated)
            advantage = td_target - curr_value

            # Calculate loss.
            critic_loss = 0.5 * (advantage**2.0)
            actor_loss = -1.0 * action_probs[action].log() * advantage.item()
            loss = critic_loss + actor_loss

            # Optimize.
            optim.zero_grad()
            loss.backward()
            optim.step()

            state = next_state
            if terminated or truncated:
                break

        # Logging.
        print(f"Episode {episode}: reward={episode_reward}")
        summary_writer.add_scalar("reward", episode_reward, episode)
        summary_writer.add_scalar("critic_loss", critic_loss.item(), episode)
        summary_writer.add_scalar("actor_loss", actor_loss.item(), episode)

    summary_writer.close()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor_critic = ActorCritic(n_obs, n_actions, N_HIDDEN_LAYERS, N_NEURONS)
    optim = Adam(actor_critic.parameters(), lr=LR)

    train(actor_critic, optim, env, TRAIN_EPISODES)
