import numpy as np
from collections import deque
from keras.models import Sequential
from skimage import transform
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
import retro

# HYPERPARAMETERS
gamma = 0.98
state_width = 84 # resized state width
state_height = 84 # resized state height
stack_size = 4 # how many states/frames to stack together to detect motion


#%%

def advantage(episode_rewards, gamma):
    discounted_rewards = np.zeros_like(episode_rewards, dtype=np.float64)
    sum = 0
    idx = len(episode_rewards) - 1
    
    for reward in reversed(episode_rewards):
        sum = sum * gamma + reward
        discounted_rewards[idx] = sum
        idx = idx - 1
    
    # normalize the discounted_rewards vector
    mean = discounted_rewards.mean()
    std = discounted_rewards.std()
    discounted_rewards = (discounted_rewards - mean) / std
    
    return discounted_rewards

def _preprocess(state):
    # convert to grayscale from rgb
    r, g, b = state[:,:,0], state[:,:,1], state[:,:,2]
    grayscale = 0.299 * r + 0.5870 * g + 0.114 * b
    
    # normalize
    grayscale /= 255.0
    
    # resize 
    grayscale = transform.resize(grayscale, (state_height, state_width))
    return grayscale

def _CNN():
    
    # CNN architecture: similar DeepMind's implementation for Atari games, except we do not downsample as aggressively
    # Input image:      84 x 84 x 4 (4 gray-scale images of 84 x 84 pixels). Input image is resized from (224, 320, 3)
    # Conv layer 1:     16 filters 3 x 3, stride 1, relu.
    # Conv layer 2:     32 filters 3 x 3, stride 1, relu.
    # Conv layer 3:     64 filters 3 x 3, stride 1, relu.
    # Fully-conn. 1:    512 units, relu.
    # Fully-conn. 2:    256 units, sigmoid. (we used sigmoid here because the game seems to allow for simultaneous actions)
    # for example, a possible action would be [0,1,1,0,1,1,1,0,0,0,1,0] 
    # Fully-conn. 2:    num-action units (12,)
    # Given the output, we choose actions with probability > 0.5\
    # loss: sum(Advantage_func * log(logits))
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu',
                     input_shape=(state_height, state_width, stack_size)))
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(12, activation='sigmoid'))
    
    return model

def _stack_frames(frames, new_state):
    # stack sequential frames together to detect motion
    
    if type(frames) != deque:
        frames = deque(frames, maxlen=stack_size)
        
    if len(frames) == 0: # having zero frames indicates that this is a new episode
        for _ in range(frames):
            frames.append(new_state) # initialize the deque with 4 copies of the first state
        stacked_states = np.stack(frames, axis=2) 
    
    frames.append(new_state)
    stacked_states = np.stack(frames, axis=2)
    
    return stacked_states, frames

class Agent():
    def __init__(self, training, level, render=False):
        self.env = retro.make(game='Airstriker-Genesis', state=level)
        self.training = training
        self.model = _CNN()
    
    def train(self, state, action):
        return
    
    def predict(self, state): # predicts an action given state
        return
    
    def preprocess(self, state): # convert to grayscale
        state = _preprocess(state)
        return state
    
    def stack_frames(self, frames, new_state):
        stacked_states = _stack_frames(frames, new_state)
        return stacked_states, new_state

    def log(self):
        # log (for results/discussion)
        # - episode number
        # - state_num
        # - life_remain
        # - episode_reward
        # - loss
        return
    
        







