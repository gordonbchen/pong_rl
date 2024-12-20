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

## Unit 4: Policy Gradients
* Parameterized stochastic policy: state -> neural net -> probability distribution over actions
* Training Loop
  * Play episode with policy
  * Calculate return (sum of rewards)
  * Update policy weights
    * Positive return -> increase probability of every state action pair
    * Negative return -> decrease probability of every state action pair
* Objective function = expected return (cumulative discounted reward)
  * sum(probability of trajectory * return of trajectory)
  * Return(trajectory) = sum((discount_factor ** n) * reward(n))
  * Prob of trajectory = product(P(s(t+1) | s(t), a(t)) * policy(a(t) | s(t)))
    * P(s(t+1) | s(t), a(t)): environmental dynamics, prob of transitioning from s(t) -> s(t + 1) given action a(t)
    * policy(a(t) | s(t)): probability dist (given by policy net) of taking a(t) given state s(t)
  * Maximize objective function by finding best parameters of policy net
* Problem with calculating derivative of J (objective function)
  * Can't calculate probability of every trajectory (sample some trajectories instead)
  * Can't differentiate state distribution (environmental dynamics)
* Policy Gradient derivation (ChatGPT is very good at explaining step by step the derivation and also meanings and interpretations of the equations)
  * $J(\pi) = E[R(\tau)] = \sum(P(\tau|\pi)R(\tau))$
  * $\frac{d}{d\theta}J(\pi) = \frac{d}{d\theta}\sum(P(\tau|\pi)R(\tau)) = \sum(R(\tau)(\frac{d}{d\theta}P(\tau|\pi)))$
  * $\frac{d}{d\theta}P(\tau|\pi) = \frac{dP(\tau|\pi)}{d\theta} * \frac{P(\tau|\pi)}{P(\tau|\pi)} = \frac{1}{P(\tau|\pi)} * \frac{dP(\tau|\pi)}{d\theta} * P(\tau|\pi) = \frac{d}{d\theta}log(P(\tau|\pi)) * P(\tau|\pi)$
  * $\frac{d}{d\theta}J(\pi) = \sum(R(\tau)(\frac{d}{d\theta}log(P(\tau|\pi)) * P(\tau|\pi)))) = E[R(\tau)\frac{d}{d\theta}log(P(\tau|\pi))]$
  * $P(\tau|\pi) = \prod(P(s_{t+1}|s_{t})\pi(a_{t}|s_{t})))$
    * $log(P(\tau|\pi)) = \sum(log(P(s_{t+1}|s_{t}))+log(\pi(a_{t}|s_{t})))) = \sum(log(P(s_{t+1}|s_{t}))) + \sum(log(\pi(a_{t}|s_{t})))$
    * $\frac{d}{d\theta}log(P(\tau|\pi)) = \frac{d}{d\theta}\sum(log(\pi(a_{t}|s_{t}))) = \sum(\frac{d}{d\theta}log(\pi(a_{t}|s_{t})))$
  * $\frac{d}{d\theta}J(\pi) = E[R(\tau)\sum(\frac{d}{d\theta}log(\pi(a_{t}|s_{t}))] = E[\sum (R(\tau) * \frac{d}{d\theta}log(\pi(a_{t}|s_{t}))]$
  * $\nabla_\theta J(\pi) = E[\sum (R(\tau) * \nabla_\theta log(\pi(a_{t}|s_{t}))]$
    * $\nabla_\theta J(\pi)$ = Gradient of objective function (direction to push params $\theta$ for steepest increase in $J(\pi)$)
    * $E$ = expectation over trajectories sampled from policy
    * $\nabla_\theta log(\pi(a_{t}|s_{t})$ means step to increase $log(\pi(a_{t}|s_{t})$. Use $log(\pi(a_{t}|s_{t})$ instead of just $\pi(a_{t}|s_{t})$ b/c strong increase ($\frac{1}{p}$) when p is low. 
    * $R(\tau) > 0$: go with direction of $\nabla_\theta log(\pi(a_{t}|s_{t})$ b/c reward is positive. If the reward is positive, push the probability of selecting that action up.
* Reinforce algorithm (Monte Carlo Reinforce)
  * Collect episode with policy
  * Use episode to estimate $\nabla_\theta J(\theta)\approx \hat{g} = \sum(R(\tau) * \nabla_\theta log\pi(a_{t}|s_{t}))$
    * $\nabla_\theta log\pi(a_{t}|s_{t})$: step of $\theta$ to increase log probability of taking action
    * $R(\tau)$: if negative, will make $\theta$ step in direction that decreases log probability of taking action, in order to make the objective function increase.
  * Update policy weights: $\theta = \theta + \alpha\hat{g}$
  * Estimating wiht multiple trajectories (episodes): $\nabla_\theta J(\theta)\approx \hat{g} = \frac{1}{m} \sum_{1}^{m} \sum(R(\tau) * \nabla_\theta log\pi(a_{t}|s_{t}))$
    * Averaging gradients across multiple episodes

## Unit 5: Advantage Actor-Critic (A2C)
* Policy-gradients
  * $\nabla_\theta J(\theta) = \sum \nabla_\theta log\pi_\theta(a_t, s_t) R(\tau)$
  * Update theta in direction that increases log prob of taking action given state if positive return
  * Return calculated from Monte-Carlo sampling, calculate cumulative discounted reward
  * Unbiased, we use the true return, not an estimation
  * High variance b/c many different trajectories can branch off at the same action, many different reward signals for the same action
  * Can mitigate variance by using many trajectories at the same time
* Actor-Critic (A2C):
  * Actor: policy function controls agent actions, $\pi_\theta(s)$
  * Critic: value function measures how good the action is $\hat{q}_w(s, a)$
  * Training loop
    * At timestep $t$, get current state $S_t$
    * Get action from policy: $A_t = \pi_\theta(S_t)$
    * Get critic value of action: $Q_{(A_t, S_t)} = \hat{q}_w(S_t, A_t)$
    * Perform action $A_t$, get new state $S_{t+1}$ and reward $R_{t+1}$
    * Update policy using q-value as return: $\Delta \theta = \alpha \nabla_\theta(log\pi_\theta(s, a)) * \hat{q}_w(s, a)$
      * Update theta in direction of increasing log prob if q-value of action, state is positive
    * Get next action from policy: $A_{t+1} = \pi_\theta(S_{t+1})$
    * Update critic parameters using TD: $\Delta w = \beta (R(s, a) + \gamma \hat{q_w}(s_{t+1}, a_{t+1}) - \hat{q_w}(s_t, a_t)) \nabla_w \hat{q_w}(s_t, a_t)$ 
      * td error: $R(s, a) + \gamma \hat{q_w}(s_{t+1}, a_{t+1}) - \hat{q_w}(s_t, a_t)$
        * difference b/t reward and discounted q-value of the next state and action, and estimated q-value at the current state
      * Update params $w$ in direction of gradient of value function if td error is positive else opposite
    * Policy and value functions have learning rates $\alpha$ and $\beta$
* Advantage
  * Advantage function: $A(s, a) = Q(s, a) - V(s)$
  * How much better is taking an action at a state better than the average value of the state
  * $A(s, a) = r + \gamma V(s') - V(s)$
  * Advantage = td error of value function = reward + discounted value of next state - value of this state
  * Instead of having state-action value function Q, have state value function V
  * Training loop
    * At timestep $t$, get current state $s$
    * Get action from policy: $a = \pi_\theta(s)$
    * Perform action $a$, get new state $s'$ and reward $r$
    * Get advantage from critic: $A(s, a) = r + \gamma V(s') - V(s)$
    * Update policy: $\Delta \theta = \alpha \nabla_\theta(log\pi_\theta(s, a)) * A(s, a)$
    * Update critic parameters using TD: $\Delta w = \beta (r + \gamma V(s') - V(s)) \nabla_w V(s)$  

## Unit 8: PPO
* Intuition behind PPO: avoid large catastrophic updates by clipping
* A2C loss: $L = L_{actor} + L_{critic}$
  * $L_{actor} = - log(p(a|s)) * A$, increase probabilities when positive advantage
  * $L_{critic} = \frac{1}{2} A^2$, push value of state closer to more informed value of next state
* PPO loss: $L = L_{actor} + L_{critic} + L_{entropy}$
  * $L_{actor} = - min(r(\theta) * A, clip(r(\theta), 1 \pm \epsilon) * A)$
    * $r(\theta) = \frac{p(a|s)}{p_{old}(a|s)}$
    * $\epsilon \approx 0.2$ is a good starting point
    * Increase probabilities when positive advantage
    * No gradient when the advantage is positive and the ratio is too high
  * $L_{critic} = \frac{1}{2} A^2$, push value of state closer to more informed value of next state
  * $L_{entropy} = \alpha * -\sum -p(x)log(p(x))$, reward model for exploration
    * $\alpha \approx 0.01$ is a good staring point
* GAE (Generalized Advantage Estimation)
  * TD: $A = \delta = (r + \gamma * V(s')) - V(s)$
    * Unbiased: local estimate of advantage
    * High variance: very short term
  * Monte Carlo Return: $A = \sum \gamma ^ t r_t$
    * More biased: depends on actual stochastic rewards accumulated during episode (possibly non-representative)
    * Lower variance: takes into account the entire episode, small fluctuations average out
  * GAE: $ A_t = \sum (\lambda \gamma)^i * \delta_{t + i}$
    * Combines benefits of TD and Monte Carlo Returns
      * $\lambda = 0$: TD
      * $\lambda = 1$: Monte Carlo
      * Lower $\lambda$ = lower bias, more variance
      * Higher $\lambda$ = higher bias, less variance
    * Parameterized bias-variance tradeoff
    * Weighted sum over TD errors
    * $\lambda \in [0.9, 0.95]$ is a good starting point
* Training loop
  * for episode in n_episodes:
    * Rollout episode: collect state, action, action prob, reward
    * for epoch in n_epochs:
      * Calculate ppo loss
        * Probability ratio
          * Recalculate probabilities for each action
          * Compare with previous action probs
        * Entropy: calculate average entropy for over all new action probability distributions
        * Advantage: calculate advantages for each state using GAE
      * Backprop and optimize
