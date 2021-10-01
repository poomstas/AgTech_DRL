# %%
# python PCSE.py
# %%
import time
import spwk_agtech
import gym
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse
import argparse
from ddpg import Agent
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# %%
assert os.environ['CONDA_DEFAULT_ENV'] == 'spacewalk', 'Switch to correct environment!'

# %%
# Parse Hyperparameter Arguments
parser = argparse.ArgumentParser(description='Hyperparameters for DDPG')
parser.add_argument('--alpha',          type=float, default=0.000025,   help='Learning Rate for the Actor (float)')
parser.add_argument('--beta',           type=float, default=0.00025,    help='Learning Rate for the Critic (float')
parser.add_argument('--input_dims',     type=list,  default=[11],       help='Dimension of state vector') 
parser.add_argument('--tau',            type=float, default=0.001,      help='Param. that allows updating of target network to gradually approach the evaluation networks. For nice slow convergence.')
parser.add_argument('--batch_size',     type=int,   default=64,         help='Batch Size for Actor & Critic training')
parser.add_argument('--layer1_size',    type=int,   default=400,        help='Layer 1 size')
parser.add_argument('--layer2_size',    type=int,   default=300,        help='Layer 2 size')
parser.add_argument('--TB_note',        type=str,   default="",         help='Note on TensorBoard')

args = parser.parse_args()

# Reference values like so: args.alpha
print()

# %%
writer_name = \
    "DDPG_alpha_{}_beta_{}_inputdims_{}_tau_{}_batchsize_{}_layer1size_{}_layer2size_{}_{}".format(
        args.alpha, args.beta, args.input_dims, args.tau, args.batch_size, 
        args.layer1_size, args.layer2_size, datetime.now().strftime("%Y%m%d_%H%M")
    )
if args.TB_note != "":
    writer_name += "_" + args.TB_note

writer_name = './TB/' + writer_name

print('TensorBoard Name: {}'.format(writer_name))
writer_path = './TB/'
writer = SummaryWriter(writer_name)

# %%
def ddpg_train(args, writer):
    ''' args contains the arguments, the writer is TensorBoard SummaryWriter object. '''

    env = gym.make('PCSE-v0')

    agent = Agent(alpha=args.alpha, beta=args.beta, input_dims=args.input_dims, tau=args.tau,
                    env=env, batch_size=args.batch_size, layer1_size=args.layer1_size,
                    layer2_size=args.layer2_size, n_actions=13)

    # Set a bunch of seeds here for reproducibility.
    np.random.seed(0)

    score_history = [] # Change name to NPV_history
    mean_score_history = [] # Change name to NPV_history; Average of the last 100 values
    
    start_time = time.time()

    for i in range(10000):
        done = False
        score = 0 # Change name to NPV
        obs = env.reset()

        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, _ = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn() # Learn at every observation, not at the end of every episode
            score += reward
            obs = new_state

        score_history.append(score)
        writer.add_scalar('episode_score', score, i)
        avg_score = np.mean(score_history[-100:])
        mean_score_history.append(avg_score)
        writer.add_scalar('last_100_score', avg_score, i)

        print('Episode: {}\tScore: {:.2f}\t\tLast 100-Trial Avg.: {:.2f}'.format(i, score, avg_score))

        if i % 25 == 0 and i != 0:
            agent.save_models()
            print("Time Since Last Save: {:.1f} sec".format(time.time() - start_time))
            start_time = time.time()

# %%
def ddpg_load_and_run():
    env = gym.make('PCSE-v0')

    # Parameter values here don't matter; will load from file.
    agent = Agent(
        alpha=1, beta=1, input_dims=[11], tau=1, env=env, batch_size=123, 
        layer1_size=123, layer2_size=123, n_actions=13
    )
    agent.load_models()
    
    while True: # Demonstrate infinietly
        done = False
        obs = env.reset()
        while not done:
            act = agent.choose_action(obs)
            obs, _, done, _ = env.step(act)
        env.render() # Render when episode is complete.

if __name__=='__main__':
    ddpg_train(args, writer)
    # ddpg_load_and_run()



# %%
# env = gym.make("PCSE-v0")

# rewards = 0
# state = env.reset()
# step = 0
# done = False
# while not done:
#     step += 1
#     action = env.action_space.sample()
#     state_, reward, done, _ = env.step(action)
#     rewards += reward
#     state = state_
# env.render()

# %%
