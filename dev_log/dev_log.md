# 11/27/2024
* Added tensorboard logging to PPO
* Did a tiny bit of undisciplined hyperparameter tuning (entropy and critic loss coefficients). There is A LOT to potentially tune though. Wondering if moving on to more advanced concepts or continuing to actually experiment (not just plug in things, but systematically tune with multiple random seeds and hyperparameter search) would be more beneficial.
* Probably going to move on and continue learning about different algorithms. But probably will come back to PPO because as far as I know, it is the state of the art in RL currently.

## 11/26/2024
* Read a lot on PPO and GAE. Started PPO implementation
* TODO: Check everything!
* Split optimization into mini-batches?
* Roll out # steps instead of a single episode? 

### 11/13/2024
* Ran A2C training again and training diverged again
* Added tensorboard logging to A2C
* Experimented with different LR values, and even tried grad clipping, didn't work
* I've never used gradient clipping successfully lol. So I'll definitely want to explore that later on.
* Updated critic loss from advantage^2 to 0.5 * (advantage^2). Completely escaped my mind how the derivative would be 2 * advantage without the half scaling.
* Merged the actor and critic models into a single model with a shared backbone and 2 outputs. This makes a lot of sense, feature extraction should generally be pretty similar between the two, and only the model head should be trained individually.
* A2C behaves much more nicely now. Still sees some training instability though.
* Likely will move on to PPO after merging this branch.
* There is still more to explore with A2C though, especially a variant called soft actor-critic. Also learned that there is A3C lol, asynchronous A2C that is distributed. Will want to explore distributed RL training and multi-agent as well later on.

### 11/11/2024
* Read and implemented the A2C algorithm.
* Added notes and equations, as well as a training script for A2C on cartpole.
* I think there are still some more complex topics in A2C to explore.
* Didn't really want to port the policy gradient training script to pong.
* Will probably save final rl port to pong for PPO.
* Next step is to continue exploring advanced A2C concepts, or start learning about PPO. 

### 10/27/2024
* HyperParam class that can be overridden through cli
* Refactored general policy gradient code out of cartpole training script (in preparation of training pong policy gradients)
* Next step is writing a script to train a pong policy gradient model 

### 10/25/2024
* Finished code for training policy gradient model on cartpole. Next step is to refactor, add hyperparams that can be overridden via cli, and then train on pong (will have to support state history stacking).
* Policy gradients are very nice to implement (I think). Simpler than double dqn when you have to train and use 2 models and keep them up to date with each other.

### 10/23/2024
* Started working on policy gradients. A lot of math today and understanding the policy gradient theorem (see derivation and interpretation in notes). I like that I can follow the math going on now (yay multivariable calculus and linear algebra)! Began working on training code for policy gradients.

### 10/15/2024
* Really want to reproduce results from https://github.com/KaleabTessera/DQN-Atari though. Very good results: increased final linear size, decreased batch size and increased lr, made min and max epsilon more extreme with more time using low epsilon, decreased replay memory size. Would want to experiment more with which hyperparams actually made the difference. But I think it is time to move on and implement policy gradients. If I have time in the end I would love to come back and see what params actually are making the difference.
* Added cli overriding for HyperParams. Started a bunch of sequential runs (overnight) using tiny bash script :). 

### 10/14/2024
* Set off training run for 1_000 episodes, with increased lr, batch size, model size, and replay memory size. Decreased epsilon decay to match increase in episodes. Slower training but reached highest performance.
* Set off training run fo 1k episodes, only with increased lr and model size. No epsilon changes. Reached highest performance (reward of 5.0 after 8.4 hrs). Beat the other computer. Good enough. Now going to focus on policy gradients.

### 10/13/2024
* Used kernprof to profile training (5 episodes).
    * Before state_history: 65.27s
    * State history = 1: 51.21s
    * State history = 2: 111.00s
    * State history = 4: 164.00s
* Biggest slowdowns: state and next_state stacking
* Fixed slowdowns by storing states and next states already moved to cuda
    * New faster training time (4): 24.79s
* Problem: store 50_000 examples, 160x210x4 states, x2 for next state as well, and 4 bytes per float32 = 53.76GB. So going to set off a training run before I go to bed. But I expect an OOM error. If I don't get one then something is wrong. YUP. Got OOM error.
* Possible solutions:
    * Store preprocessed frames: 50_000 * 2 * 4x32x32 * 4 = 1.64GB.
    * Dataloader storing on cpu then moving to gpu with multiprocessing.
* Implemented storing preprocessed states instead of full frames: 20.50s.
* Set off training run for 500 episodes.

### 10/12/2024
* Training now takes more than 2x longer.
* Possible reasons for 2x slowdown:
    * Conv input size for getting action is now 4x32x32 (4 frames of state history) instead of 1x32x32. (No solution here.)
    * Deque in training loop, with np.stack happening 2x per step (to get state and next_state from obs hsitory).
    * Maybe move each state to cuda when discovered in training loop
        * Benefit from faster tensor ops (hopefully faster than np.stack) and no moving to cuda during training step
        * Disadvantages: many more slower individual moves to gpu, also will want to see if we have enough memory for this.
    * Will want to benchmark with kernprof
* Thinking about increasing learning rate and overall HyperParams, want to balance HyperParam tuning with continuing with Policy Gradients though.
* Increased black and flake8 max line length to 100. More readable I think.

### 10/11/2024
* Created dev log for logging experiments, setbacks, ideas, and progress.
* Progress
    * save HyperParams to json: easier to track experiments
    * Checked resizing pong frames to 32x32. Playable, also checked 48x48 and 64x64.
    * Realized that feeding frame diffs to network is bad b/c if the agent doesn't move, you can't tell where your own paddle is. Switching to feeding 4 frames instead.
    * Set off run with only change being feeding state history instead of state diff.
* Ideas
    * Create ReplayMemory class with data loading multiprocessing queue
    
