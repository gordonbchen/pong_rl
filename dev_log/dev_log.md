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
* ![10/12/2024 State History run reward plot](dev_log_images/10_12_2024_state_history.png)
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
    
