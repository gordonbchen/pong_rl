10/11/2024
* Created dev log for logging experiments, setbacks, ideas, and progress.
* Progress
    * save HyperParams to json: easier to track experiments
    * Checked resizing pong frames to 32x32. Playable, also checked 48x48 and 64x64.
    * Realized that feeding frame diffs to network is bad b/c if the agent doesn't move, you can't tell where your own paddle is. Switching to feeding 4 frames instead.
* Ideas
    * Create ReplayMemory class with data loading multiprocessing queue
    
