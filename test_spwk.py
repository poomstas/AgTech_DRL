import spwk_agtech
import gym

env = gym.make("PCSE-v0") # there is just version 0 in now

rews = 0
obs = env.reset()
step = 0
while True:
    step += 1
    action = env.action_space.sample()
    next_obs, rew, done, _  = env.step(action)
    rews += rew
    obs = next_obs
    if done:
        break

