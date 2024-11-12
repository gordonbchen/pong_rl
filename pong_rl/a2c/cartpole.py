"""Train an Advantage Actor Critic (A2C) model on cartpole."""

import gymnasium as gym

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.optimizer import Optimizer


LR = 3e-5
GAMMA = 0.99
TRAIN_EPISODES = 1_000
N_HIDDEN_LAYERS = 4
N_NEURONS = 256


class MLP(nn.Module):
    """A multi-layer perceptron using ReLU activation."""

    def __init__(self, n_inputs: int, n_outputs: int, n_hidden_layers: int, n_neurons: int) -> None:
        """Init actor model."""
        super().__init__()

        in_layer = nn.Sequential(nn.Linear(n_inputs, n_neurons), nn.ReLU())
        hidden_layers = nn.Sequential(
            *[nn.Linear(n_neurons, n_neurons), nn.ReLU()] * (n_hidden_layers - 1)
        )
        out_layer = nn.Linear(n_neurons, n_outputs)
        self.net = nn.Sequential(in_layer, hidden_layers, out_layer)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Returns logits."""
        return self.net(xb)


def train(
    actor: nn.Module,
    critic: nn.Module,
    actor_optim: Optimizer,
    critic_optim: Optimizer,
    env: gym.Env,
    train_episodes: int,
):
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
    * Update value model: loss = advantage ** 2
    """
    actor.train().to("cuda")
    critic.train().to("cuda")

    for episode in range(train_episodes):
        episode_reward = 0

        obs, _ = env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device="cuda").unsqueeze(0)
        while True:
            # Get action from actor.
            action_probs = actor(state).squeeze(0)
            action = torch.multinomial(action_probs, 1).item()

            # Take action.
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            next_state = torch.tensor(next_obs, dtype=torch.float32, device="cuda").unsqueeze(0)

            # Calculate advantage.
            # NOTE: 0 when truncated as well?
            with torch.no_grad():
                td_target = reward + (GAMMA * critic(next_state) * (1 - terminated))
            advantage = td_target - critic(state)

            # Optimize critic.
            critic_loss = advantage**2.0
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # Optimize actor.
            actor_loss = -1.0 * action_probs[action].log() * advantage.detach()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            state = next_state
            if terminated or truncated:
                break

        print(f"Episode {episode}: reward={episode_reward}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    print(f"n_obs={n_obs}, n_actions={n_actions}")

    actor = nn.Sequential(
        MLP(
            n_inputs=n_obs,
            n_outputs=n_actions,
            n_hidden_layers=N_HIDDEN_LAYERS,
            n_neurons=N_NEURONS,
        ),
        nn.Softmax(dim=-1),
    )
    print(actor)

    critic = MLP(n_inputs=n_obs, n_outputs=1, n_hidden_layers=N_HIDDEN_LAYERS, n_neurons=N_NEURONS)
    print(critic)

    actor_optim = Adam(actor.parameters(), lr=LR)
    critic_optim = Adam(critic.parameters(), lr=LR)

    train(actor, critic, actor_optim, critic_optim, env, train_episodes=TRAIN_EPISODES)
