# Load the best performing action and visualize the results
# %%
import spwk_agtech
import gym
import pickle

actions_file = 'best_action_save.pickle'

with open(actions_file, 'rb') as f:
    actions = pickle.load(f)

# %%
env = gym.make('PCSE-v0')

obs = env.reset()
reward_sum = 0
step = 0

for action in actions:
    obs_, reward, done, _ = env.step(action)

env.render()

# %%
