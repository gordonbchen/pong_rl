# pong_rl
Training a reinforcement learning model to play pong (UConn CSE2050).
* Q-Table
* DQN (Deep Q Network): implementation for cartpole and pong (all others models only have cartpole training code)
* Policy Gradients (Reinforce)
* A2C (Advantage Actor Critic)
* PPO (Proximal Policy Optimization) with GAE (Generalized Advantage Estimation)

## Files
* `dev_log`: contains RL notes, project proposal, and project development log
* `pong_rl`: contains all model training scripts

## Requirements
* Requires `python3` and `poetry`
* Install dependencies with `poetry install`, and activate poetry env with `poetry shell`
* Notable dependencies
    * `pytorch` for autodiff and nn
    * `gymnasium` for cartpole and pong rl environments
    * `tensorboard` for logging

## Usage
* Run `python3 pong_rl/[model type]/[game training script]`
    * Ex: `python3 pong_rl/ppo/cartpole.py`
    * Trains an rl model to play the game
    * Training outputs and `tensorboard` logs will be put in `outputs`
* Launch `tensorboard` to inspect training

## Sources
* HuggingFace RL Tutorial: https://huggingface.co/learn/deep-rl-course/unit0/introduction
* PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
* PyTorch Lightning DQN Tutorial: https://lightning.ai/docs/pytorch/LTS/notebooks/lightning_examples/reinforce-learning-DQN.html
* cleanrl DQN implementation: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py#L107
* KaleabTessera DQN implementation: https://github.com/KaleabTessera/DQN-Atari
* Deep Reinforcement Learning: Pong from Pixels by Andrej Karpathy: https://karpathy.github.io/2016/05/31/rl/
* A2C article: https://medium.com/@dixitaniket76/advantage-actor-critic-a2c-algorithm-explained-and-implemented-in-pytorch-dc3354b60b50
* OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/sac.html
* Costa Huang + CleanRL PPO: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
