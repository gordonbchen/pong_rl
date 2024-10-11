# Pong RL

## Description
I plan to train a reinforcement learning model to play the game of pong using pong's raw frames. The reinforcement learning model (a deep neural network of some kind, probably a conv net) will output probabilities for moving up, down, or staying. The model will be trained by rewarding the model when it wins and punishing it when it loses using policy gradients. I will use `pytorch` to train the neural net.

The goal is for our model to beat the built-in computer player provided by `gymnasium`, and hopefully even beat us when we play against it.

This project will require me to learn a lot about the techniques of reinforcement learning to complete.

## Plan and Deliverables
* Read and learn a lot about reinforcement learning (specifically policy gradients).
* **Deliverable 1 (9/16)**: notes about RL and basic psuedocode for training pong rl model
* Implement and train pong rl model
* **Deliverable 2 (10/28)**: base working code for training pong rl model, report on challenges and solutions
* Fine tune model and training code, experiment with model, different preprocessing
* **Deliverable 3 (11/18)**: fine-tuned model, pong demo, model weights
* Reflect on project, cementing and distilling knowledge into explanation
* **Deliverable 3 (11/29)**: explanation of how to train pong rl model
