# %%
import time
import gym
import numpy as np
import argparse

from datetime import datetime
from td3_torch import Agent
from torch.utils.tensorboard import SummaryWriter

# %%
def parse_arguments(parser):
    parser.add_argument('--alpha',          type=float, default=0.001,  help='Learning Rate for the Actor (float)')
    parser.add_argument('--beta',           type=float, default=0.001,  help='Learning Rate for the Critic (float')
    parser.add_argument('--tau',            type=float, default=0.005,  help='Controls soft updating the target network')
    parser.add_argument('--batch_size',     type=int,   default=100,    help='Batch Size for Actor & Critic training')
    parser.add_argument('--layer1_size',    type=int,   default=400,    help='Layer 1 size (same for actor & critic)')
    parser.add_argument('--layer2_size',    type=int,   default=300,    help='Layer 2 size (same for actor & critic)')
    parser.add_argument('--n_games',        type=int,   default=10000,  help='Total number of episodes')
    parser.add_argument('--TB_note',        type=str,   default="",     help='Note on TensorBoard')

    args = parser.parse_args()

    return args

# %%
def get_writer_name(args):
    writer_name = \
        "TD3LunarLanderCont_alpha_{}_beta_{}_tau_{}_batchsize_{}_layer1size_{}_layer2size_{}_nGames_{}_{}".format(
            args.alpha, args.beta, args.tau, args.batch_size, args.layer1_size, args.layer2_size, 
            args.n_games, datetime.now().strftime("%Y%m%d_%H%M")
        )

    if args.TB_note != "":
        writer_name += "_" + args.TB_note # TB_note is handled separately

    writer_name = './TB/' + writer_name
    print('TensorBoard Name: {}'.format(writer_name))

    return writer_name

# %%
def train_td3(args, writer):
    train_begin_time = time.time()

    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=args.alpha, beta=args.beta, input_dims=env.observation_space.shape,
                    tau=args.tau, env=env, batch_size=args.batch_size, layer1_size=args.layer1_size, 
                    layer2_size=args.layer2_size, n_actions=env.action_space.shape[0])

    best_score = env.reward_range[0]
    score_history = []
    
    for i in range(args.n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        writer.add_scalar('Score', score, i)
        avg_score = np.mean(score_history[-100:])
        writer.add_scalar('last_100_score_avg', avg_score, i)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        writer.add_scalar('best_score_so_far', best_score, i)

        if i % 20 == 0:
            print('Episode: {:<6s}\tScore: {:<10s}\tLast 100 Episode Avg.: {:<15s}\tTrain Time: {:.1f} sec'.format(
                str(i), str(np.round(score, 2)), str(np.round(avg_score, 2)), time.time()-train_begin_time))

# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters for TD3') # Parse hyperparameter arguments from CLI
    args = parse_arguments(parser) # Reference values like so: args.alpha 

    writer_name = get_writer_name(args)
    writer = SummaryWriter(writer_name)

    train_td3(args, writer)
