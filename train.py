import agent
import numpy as np

# train with openAI gym retro

# =============================================================================
# pseudo code:
#     Create the NN 
#     maxReward = 0
#     Keep track of maximum reward 
#     For episode_num in range(max_episodes): # max_episodes is the rollout
#         episode + 1; reset environment; reset stores (states, actions, rewards) 
#         For each step: 
#             Choose action a 
#             Perform action a - always instruct aircraft to shoot; train maneuver only 
#             Store state, action (one hot vector), reward 
#             If done: 
#                 Calculate 
#                 Calculate discounted episode reward (discount_factor * reward)
#                 Optimize
# =============================================================================

max_episodes = 1000
gamma = 0.98
state_width = 84 # resized state width
state_height = 84 # resized state height
stack_size = 4 # how many states/frames to stack together to detect motion
frames = []

player = agent.Agent('Airstriker-Genesis', 'Level1')

# variables to log
with open('log_reward.txt', mode='a', buffering=1) as f:
    msg = 'episode_num\tmean_reward\ttotal_reward\n'
    f.write(msg)
print('Begin roll out...')

for episode_num in range(max_episodes): # or let it run indefinitely (?)
    
    frames= []
    all_states = []
    all_actions = []
    rewards = []
    total_reward = 0
    lives = 3
    done = False
    state = player.reset()
    old_frames = None
    
    while not done:
        state = player.preprocess(state)
        frames = player.stack_frames(frames, state)
        stacked_states = np.stack(frames, axis=2) # dimension of stacked_states: (84, 84, 4); stacked frames necessary to detect motion
        action = player.get_action(stacked_states) # get an action given state
        new_state, r, done, info = player.perform_action(action)
        total_reward += r
        all_states.append(stacked_states)
        all_actions.append(action)
        state = new_state
        
        # clip the rewards and penalize if agent loses a life
        if info['lives'] == lives:
            if r > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    
        if info['lives'] < lives:
            rewards.append(-1.0)
            lives = info['lives']
        
        if done:
            # record episode number and the mean reward for this episode
            with open('log_reward.txt', mode='a', buffering=1) as f:
                mean_reward = total_reward / len(rewards)
                msg = '{0}\t{1:.2f}\t{2}\n'.format(episode_num+1, mean_reward, total_reward)
                print('Episode: {0}\tMean Reward: {1:.2f}\tTotal Reward: {2}\n'.format(episode_num+1, mean_reward, total_reward))
                f.write(msg)
                
            episode_states = np.array(all_states)
            episode_actions = np.vstack(np.array(all_actions))
            episode_rewards = np.vstack(np.array(rewards))
            episode_rewards = agent.discount(episode_rewards, gamma)
            
            # error occurs when the episode plays for too long; not sure why
            # clip the states, actions and rewards if the episode exceeds 3000 states
            if len(episode_states) > 3000:
                episode_states = episode_states[:3000]
                episode_actions = episode_actions[:3000]
                episode_rewards = episode_rewards[:3000]
            player.train(episode_states, episode_actions, episode_rewards)

# Model summary        
# =============================================================================
#             print(episode_states.shape) # (num_states, 84, 84, 4)
#             print(episode_actions.shape) # (num_states, 12)
#             print(episode_rewards.shape) # (num_states, 1)
#             print(player.model.summary()) 
# =============================================================================
        
    # save weights once in a while
    if episode_num % 100 == 0:
        player.save()
        
    
    
    