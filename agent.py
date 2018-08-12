import numpy as np
from collections import deque
from keras.models import Model
from skimage import transform
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, MaxPooling2D
from keras import optimizers
import keras.backend as K
import retro

# HYPERPARAMETERS
state_width = 84 # resized state width
state_height = 84 # resized state height
stack_size = 4 # how many states/frames to stack together to detect motion

#%%

def discount(episode_rewards, gamma):
    discounted_rewards = np.zeros_like(episode_rewards, dtype=np.float64)
    
    # to prevent division by 0 during normalization, which may occur if the 
    # agent received no reward during an episode 
    sum = 0
    
    for t in reversed(range(0, len(episode_rewards))):
        sum = sum * gamma + episode_rewards[t]
        discounted_rewards[t] = sum

    # normalize the discounted_rewards vector; omitted due to numerical instability
# =============================================================================
#     mean = discounted_rewards.mean()
#     std = discounted_rewards.std()
#     discounted_rewards = (discounted_rewards - mean) / (std + 1e-10)
# =============================================================================
    
    return discounted_rewards

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def _preprocess(state):
    # convert to grayscale from rgb
    r, g, b = state[:,:,0], state[:,:,1], state[:,:,2]
    grayscale = 0.299 * r + 0.5870 * g + 0.114 * b
    
    # normalize
    grayscale /= 255.0
    
    # resize 
    grayscale = transform.resize(grayscale, (state_height, state_width))
    return grayscale

def _loss(actions, logits, discounted_rewards):
    cross_entropy = K.tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions, logits=logits)
    J = K.tf.reduce_mean(cross_entropy * discounted_rewards)
    return J

def _CNN(mode='train'):
    
    # CNN architecture: similar DeepMind's implementation for Atari games, except we do not downsample as aggressively
    # Input image:      84 x 84 x 4 (4 gray-scale images of 84 x 84 pixels). Input image is resized from (224, 320, 3)
    # Conv layer 1:     16 filters 3 x 3, stride 1, relu.
    # Conv layer 2:     32 filters 3 x 3, stride 1, relu.
    # Conv layer 3:     64 filters 3 x 3, stride 1, relu.
    # Fully-conn. 1:    512 units, relu.
    # Fully-conn. 2:    256 units, softmax
    # Fully-conn. 2:    num-action units (12,)
    # Given the output, we choose actions with probability > 0.5
    # loss: sum(Advantage_func * log(logits))
    
    img = Input(shape=(state_height, state_width, stack_size))
    yTrue = Input(shape=(12,))
    r = Input(shape=(1,))
    
    conv = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu',)(img)
    conv = MaxPooling2D()(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(conv)
    conv = MaxPooling2D()(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(conv)
    conv = MaxPooling2D()(conv)
    conv = BatchNormalization()(conv)
    l = Flatten()(conv)
    l = Dense(512, activation='relu')(l)
    l = Dense(256, activation='relu')(l)
    yPred = Dense(12, activation='linear')(l)
    
    trainer = Model(inputs=[img, yTrue, r], outputs=yPred)
    trainer.add_loss(_loss(yTrue, yPred, r))
    adam = optimizers.Adam(lr=0.01)
    trainer.compile(loss=None, optimizer=adam)
    
    if mode == 'train':
        return trainer
    else:
        return Model(inputs=img, outputs=yPred)

def _stack_frames(frames, new_state):
    # stack sequential frames together to detect motion
    
    if type(frames) != deque:
        frames = deque(frames, maxlen=stack_size)
        
    if len(frames) == 0: # having zero frames indicates that this is a new episode
        for _ in range(stack_size):
            frames.append(new_state) # initialize the deque with 4 copies of the first state
    else:
        frames.append(new_state)
        
    return frames

class Agent():
    def __init__(self, game, level):
        self.env = retro.make(game, level)
        self.trainer = _CNN(mode='train')          
        self.predictor = _CNN(mode='predict')
            
    def train(self, states, actions, discounted_rewards):
        self.trainer.fit([states, actions, discounted_rewards], batch_size=16, epochs=1)
    
    def preprocess(self, state): # convert to grayscale
        state = _preprocess(state)
        return state
    
    def stack_frames(self, frames, new_state):
        frames = _stack_frames(frames, new_state)
        return frames
    
    def get_action(self, state):
        logits = self.predictor.predict(state.reshape(1,state_height,state_width,stack_size))
        logits = softmax(logits)
        logits = logits.squeeze() # from (1, 12) to (12,)
        action = np.zeros(logits.shape[0])
        idx = np.random.choice(logits.shape[0], p=logits) # sample action from probability distribution  
        action[idx] = 1
        return action
    
    def perform_action(self, action, render=False):
        observation, reward, done, info = self.env.step(action)
        if render == True:
            self.env.render()
        return observation, reward, done, info
    
    def reset(self):
        state = self.env.reset()
        return state
    
    def save(self):
        self.trainer.save('./model/agent.h5')