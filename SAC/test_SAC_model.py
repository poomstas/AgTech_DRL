# %%
import os
import spwk_agtech
import gym
from tqdm import tqdm
import numpy as np

from sac_torch import Agent

assert os.environ['CONDA_DEFAULT_ENV'] == 'spacewalk', 'Switch to correct environment!'

# %%
MODEL_REFNAME = 'SAC_PCSE_alpha_0.01_beta_0.01_tau_0.1_RewScale_3_batchsize_100_layer1size_256_layer2size_256_nGames_1000_patience_500_20211016_002751_Run commands auto-generated 20211015'
MODEL_PATH     = '/home/brian/Dropbox/SideProjects/20210915 Spacewalk Interview/AgTech/SAC/TB'

# %%
env = gym.make('PCSE-v0')

# %% 
# Actual parameter values below don't matter; will be overwritten when loading from saved model. 
agent = Agent(env=env, input_dims=env.observation_space.shape, n_actions=env.action_space.shape[0],
            TB_name='TestRun', alpha=0.001, beta=0.001, tau=0.001, batch_size=100,
            layer1_size=256, layer2_size=256, reward_scale=2)

# %% 
# Run an episode to termination 100 times using the loaded agent, calculate avg profit
profits = []
for i in tqdm(range(100)):
    agent.load_models(model_refname=MODEL_REFNAME, model_path=MODEL_PATH)

    obs = env.reset()
    done = False
    reward_sum = 0
    step = 0

    while not done:
        step += 1
        action = agent.choose_action(obs)
        obs_, reward, done, _ = env.step(action)
        agent.remember(obs, action, reward, obs_, done)
        agent.learn()
        reward_sum += reward
        obs = obs_

    profits.append(env.profit)

# %%
print('Avg. Profit: {:.2f}'.format(np.mean(profits)))
print('Max. Profit: {:.2f}'.format(np.max(profits)))
