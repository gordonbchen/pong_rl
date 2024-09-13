# HuggingFace RL Course Notes

## Unit 1: Intro to Deep RL
* Recieve state S0, take action A0, go to new state S1, recieve reward R1.
* Reward hypothesis: maximize cumulative reward
* Markov Decision Process: only current state is needed to decide what action to take

* State: complete description of world, fully observed
* Observation: partial description of state

* Action space: set of all possible actions
* Discrete: finite possible actions: left, right, up, down
* Continuos: infinite possible actions: angle to turn to

* Discounted cumulative reward: sum((gamma ** n) * r(n))
* Discount because immediate rewards > future rewards (predictability and immediacy)
* Gamma on (0, 1]
* High gamma = small discount + long-term reward, (gamma ** n) stays close to 1
* Low gamma = larger discount + short-term reward, (gamma ** n) decays faster

* Episodic task: starting and terminal (ending) state
* Continuing task: no terminal state

* Exploration: exploring the environment, trying random actions in order to learn about the environment
* Exploitation: exploiting known information to maximize reward
* Want to explore much more at the start, learn about the possible rewards in the environ, and then exploit

* Policy(state) = action
* Policy-based: learn policy directly, policy(state) = probability distribution of actions

* Value-based: learn value function: value_func(state) = expected discounted reward
* Train value function, use values to select action using hand-crafted policy

## Unit 2: Intro to Q-Learning
* Policy-based: neural network is the policy, no value function, trains policy directly
* Value-based: train value function neural network, policy is hand-crafted

* State-value function: Value(state) = expected return
* Action-value function: Value(state, action) = expected return, value of starting in state and taking that action

* Bellman equation: Value(state) = expected immediate reward + discounted value of next state
* Recursive, since discounted value of the next state is calculated with the bellman equation as well
* Think about expanding discounted value of next state out

* Monte Carlo: learning at the end of an episode
* wait until the episode, calulate the discounted cumulative return (target), and update the value of the state
* new state value = former estimation + learning rate * (actual return - former estimation) 
* new state value = ((1 - learning_rate) * former estimation) + (learning rate * actual return)

* Temporal Difference: learning at each step
* wait for 1 step, update value using immediate reward and estimated value of next state
* bootstrapping: estimating G(t) (true expected return) by immediate reward and estimated value of next state
* new state value = former_estimation + learning_rate * (immediate_reward + discounted_estimate_of_next_state_value - former_estimation)

* Q-Learning: off-policy, value-based, temporal difference
* Q-function: action-value function, value of being at a state and taking an action
* Steps
  * initialize q-table
  * choose action using epsilon-greedy policy (1 - epsilon is greedy, epsilon is random)
    * start with high epsilon value, explore more, then decay over training, favoring exploitation
  * Take action At, observe reward Rt+1 and new state St+1
  * Update Q(St, At) = Q(St, At) + lr * (Rt+1 + gamma * max(Q(St+1)) - Q(St, At))
* Off-policy: different policy for acting (epsilon greedy) and updating (max value for state over all possible actions)

## Unit 3: Deep Q-Learning with Atari Games
* Q-learning is not scalable: FrozenLake = 16 states, Taxi-v3 = 500 states, Atari = (210, 160, 3) * 256 states.
* Deep Q-learning: state --network--> q-value of actions

* DQN: 4 frames --conv layers--> --fully connected layers--> q-values of (left, right, shoot)
* Then use epsilon-greedy policy to choose action during training

* Preprocess to reduce state complexity: (210, 160, 3) --downsize--> (84, 84, 4), rgb -> grayscale, crop
* Stack 4 frames to see things moving, velocity

* Q-learning: new_state_value = former_estimate + learning_rate * (immediate_reward + gamma * max_next_state_reward - former_estimate)
* Deep q-learning
  * q-target = immediate_reward + (gamma * max_estimated_next_state_rewawrd)
  * q-loss = q_target - former_estimation

* Sampling: perform actions, store experiences in replay memory
* Training: select random batches of experiences and learn using gradient descent

* Initialize replay_memory
* Epsilon-greedy: random action or argmax(dqn(state))
* Take action, store transition (state, action, reward, new_state) in replay_memory
* Sample random minibatch of transitions from D
* Target = immediate_reward + gamma * max(dqn_fixed(new_state))
  * target = immediate_reward if terminates and no new_state
* Perform gradient descent step on dqn, loss = (target - dqn(state)) ** 2
* Reset dqn_fixed = dqn every C steps

* Experience replay
  * efficiency: experiences aren't only used 1 time then thrown away
  * avoid catastrophic forgetting

* Fixed Q-target
  * if we dont fix it: both q_values and targets are being updated at the same time (same neural net)
  * moving goalpost, harder to train
  * use separate network w/ fixed params to get target, then copy params from unfixed dqn every C steps

* Double deep q-learning
  * DQN: selects best action
  * Target network (fixed DQN): calculates target q-value of the next state
