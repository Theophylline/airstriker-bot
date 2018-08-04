# train with openAI gym retro

# =============================================================================
# pseudo code:
#     Create the NN maxReward = 0
#     Keep track of maximum reward 
#     For episode_num in range(max_episodes): # max_episodes is the rollout
#         episode + 1 reset environment reset stores (states, actions, rewards) 
#         For each step: 
#             Choose action a 
#             Perform action a - always instruct aircraft to shoot; train maneuver only 
#             Store state, action (one hot vector), reward 
#             If done: 
#                 Calculate 
#                 Calculate discounted episode reward (discount_factor * reward)
#                 Optimize
# =============================================================================
