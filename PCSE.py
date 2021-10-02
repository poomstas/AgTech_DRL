# %%
import time
import spwk_agtech
import gym
import numpy as np
import time
import os
import argparse
import argparse
from ddpg import Agent
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# %%
assert os.environ['CONDA_DEFAULT_ENV'] == 'spacewalk', 'Switch to correct environment!'

# %%
def parse_arguments(parser):
    parser.add_argument('--alpha',          type=float, default=0.000025,   help='Learning Rate for the Actor (float)')
    parser.add_argument('--beta',           type=float, default=0.00025,    help='Learning Rate for the Critic (float')
    parser.add_argument('--tau',            type=float, default=0.001,      help='Param. that allows updating of target network to gradually approach the evaluation networks. For nice slow convergence.')
    parser.add_argument('--gamma',          type=float, default=0.99,       help='Discount factor')
    parser.add_argument('--batch_size',     type=int,   default=64,         help='Batch Size for Actor & Critic training')
    parser.add_argument('--layer1_size',    type=int,   default=400,        help='Layer 1 size (same for actor & critic)')
    parser.add_argument('--layer2_size',    type=int,   default=300,        help='Layer 2 size (same for actor & critic)')
    parser.add_argument('--TB_note',        type=str,   default="",         help='Note on TensorBoard')

    args = parser.parse_args()

    return args

# %%
def get_writer_name(args):
    writer_name = \
        "DDPG_alpha_{}_beta_{}_tau_{}_batchsize_{}_layer1size_{}_layer2size_{}_{}".format(
            args.alpha, args.beta, args.tau, args.batch_size, 
            args.layer1_size, args.layer2_size, datetime.now().strftime("%Y%m%d_%H%M")
        )
    if args.TB_note != "":
        writer_name += "_" + args.TB_note

    writer_name = './TB/' + writer_name
    print('TensorBoard Name: {}'.format(writer_name))

    return writer_name

# %%
def has_plateaued(reward_history, patience=100):
    ''' Simple function that checks for plateau. '''
    single_patience_mean = np.mean(reward_history[-patience:])
    double_patience_mean = np.mean(reward_history[-2*patience:])

    if len(reward_history) < 2*patience:
        return False

    plateau_bool = np.abs((single_patience_mean - double_patience_mean) / single_patience_mean)*100 < 0.1

    return plateau_bool

# %%
def ddpg_train(args, writer):
    ''' args contains the arguments, the writer is TensorBoard SummaryWriter object. '''
    train_begin_time = time.time()

    env = gym.make('PCSE-v0')

    agent = Agent(alpha=args.alpha, beta=args.beta, input_dims=[11], tau=args.tau,
                    TB_name=writer.log_dir, gamma=args.gamma,
                    batch_size=args.batch_size, layer1_size=args.layer1_size,
                    layer2_size=args.layer2_size, n_actions=13)

    # Set a bunch of seeds here for reproducibility.
    np.random.seed(0)

    reward_history = []
    mean_reward_history = [] # Average of the last 100 values
    best_mean_reward = -float('inf')
    
    start_time = time.time()

    for i in range(30000): # Approx 30,000 iter/day (will almost certainly terminate before)
        done = False
        reward_sum = 0 # Change name to NPV
        obs = env.reset()

        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, _ = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn() # Learn at every observation, not at the end of every episode -> But doesn't actually do it until the buffer is filled up.
            reward_sum += reward
            obs = new_state

        reward_history.append(reward_sum)
        writer.add_scalar('episode_reward', reward_sum, i)
        last_100_reward_avg = np.mean(reward_history[-100:])
        mean_reward_history.append(last_100_reward_avg)
        writer.add_scalar('last_100_reward_avg', last_100_reward_avg, i)

        print('Episode: {:<6s}\tReward: {:<10s}\tLast 100 Episode Avg.: {:<15s}\tTrain Time: {:.1f} sec'.format(
            str(i), str(np.round(reward_sum, 2)), str(np.round(last_100_reward_avg, 2)), time.time()-train_begin_time)
            )
        
        if has_plateaued(mean_reward_history):
            print("\nReached Plateau; Terminating Simulations.\n")
            print("Writer: {}".format(writer.log_dir))
            break

        if i % 25 == 0 and i != 0:
            if last_100_reward_avg > best_mean_reward:
                best_mean_reward = last_100_reward_avg
                agent.save_models()
                print("Time Since Last Save: {:.1f} sec".format(time.time() - start_time))
                start_time = time.time()

# %%
def ddpg_load_and_run():
    ''' 
        This needs some fixing! agent.load_models() gives the following error: 
        FileNotFoundError: [Errno 2] No such file or directory: '_Actor_ddpg'
    '''
    env = gym.make('PCSE-v0')

    # Parameter values here don't matter; will load from file.
    agent = Agent(alpha=1, beta=1, input_dims=[11], tau=1, TB_name="", 
                  batch_size=64, layer1_size=1, layer2_size=1, n_actions=13)
    agent.load_models()
    
    while True: # Demonstrate infinietly
        done = False
        obs = env.reset()
        while not done:
            act = agent.choose_action(obs)
            obs, _, done, _ = env.step(act)
        env.render() # Render when episode is complete.

# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Hyperparameters for DDPG') # Parse hyperparameter arguments from CLI
    args = parse_arguments(parser) # Reference values like so: args.alpha 

    writer_name = get_writer_name(args)
    writer = SummaryWriter(writer_name)

    ddpg_train(args, writer)
    # ddpg_load_and_run()   # This needs to be fixed!
