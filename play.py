# testing and evaluation

import retro

def main():
    env = retro.make(game='Airstriker-Genesis', state='Level1')
    state = env.reset()
    while True:  
        action = env.action_space.sample() # action dimension = (12,)
        # state dim = (224,320, 3); will be converted to grayscale and downsized
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            state = env.reset()
        
if __name__ == '__main__':
    main()
    
#%%
    

