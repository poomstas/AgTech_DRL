import time
import gym
import numpy as np
from td3_torch import Agent
# from utils import plot_learning_curve

if __name__=='__main__':
    train_begin_time = time.time()

    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape,
                    tau=0.005, env=env, batch_size=100, layer1_size=400, 
                    layer2_size=300, n_actions=env.action_space.shape[0])
    n_games = 10000
    filename = 'plots/' + 'LunarLanderContinuous_' + str(n_games) + '_games.png'

    best_score = env.reward_range[0]
    score_history = []
    
    # agent.load_models()

    for i in range(n_games):
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
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        if i % 20 == 0:
            print('Episode: {:<6s}\tScore: {:<10s}\tLast 100 Episode Avg.: {:<15s}\tTrain Time: {:.1f} sec'.format(
                str(i), str(np.round(score, 2)), str(np.round(avg_score, 2)), time.time()-train_begin_time))

        x = [i+1 for i in range(n_games)]
        # plot_learning_curve(x, scores, filename)

