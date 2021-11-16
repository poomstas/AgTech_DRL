# %%
import os
import spwk_agtech
import gym
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from sac_torch import Agent

# %%
MODEL_REFNAME = 'SAC_PCSE_alpha_0.001_beta_0.001_tau_0.01_RewScale_18_batchsize_100_layer1size_256_layer2size_256_nGames_50000_patience_1000_20211023_005715_Run commands auto-generated 20211016' 
MODEL_PATH     = '/home/brian/Dropbox/SideProjects/20210915 Spacewalk Interview/AgTech/SAC/TB'

# %%
env = gym.make('PCSE-v0')

# Actual parameter values below don't matter; will be overwritten when loading from saved model. 
agent = Agent(env=env, input_dims=env.observation_space.shape, n_actions=env.action_space.shape[0],
            TB_name='TestRun', alpha=0.001, beta=0.001, tau=0.001, batch_size=100,
            layer1_size=256, layer2_size=256, reward_scale=2)

# %% 
# Run an episode to termination 1000 times using the loaded agent, calculate avg profit
profits = []
best_reward = -float('inf')

for i in tqdm(range(1000)):
    agent.load_models(model_refname=MODEL_REFNAME, model_path=MODEL_PATH, gpu_indx=0)

    obs = env.reset()
    done = False
    reward_sum = 0
    step = 0
    current_episodes_actions = []

    while not done:
        step += 1
        action = agent.choose_action(obs)
        current_episodes_actions.append(action)
        obs_, reward, done, _ = env.step(action)
        agent.remember(obs, action, reward, obs_, done)
        agent.learn()
        reward_sum += reward
        obs = obs_
    
    if reward_sum > best_reward:
        best_reward = reward_sum
        best_action = current_episodes_actions

    profits.append(env.profit)

# %%
print('Best Reward: {:.2f}'.format(best_reward))
print('Best Action: ', best_action)

with open('best_action_save.pickle', 'wb') as f:
    pickle.dump(best_action, f)

# %%
print('Avg. Profit: {:.2f}'.format(np.mean(profits)))
print('Max. Profit: {:.2f}'.format(np.max(profits)))

plt.hist(profits)
plt.xlabel('Episode Rewards ($/ha)', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('SAC Final Model Performance', fontweight='bold')
plt.show()

# %%
