# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# %%
class OUActionNoise:
    ''' Orenstein-Uhlenbeck Noise implementation. Temporally correlated noise with mu=0.'''
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        ''' Enables the operation below:
            noise = OUActionNoise()
            noise() '''
        x = self.x_prev + self.theta * (self.mu-self.x_prev)*self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# %%
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done 
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

# %%
class CriticNetwork(nn.Module):
    ''' Beta: Learning rate '''
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='./ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0]) # No. for initializing the wts and biases of that NN layer
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims) # BatchNorm Layer

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims) # Because we have batchnorm layers, we have to use .eval() and .train() methods correctly.

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.003 # From the paper
        self.q = nn.Linear(self.fc2_dims, 1) # Scalar value, has only one output
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        ''' The action is continuous. In this case, it's np.array vector of length 13. '''
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value) # Open debate as to whether the relu should come before or after the BN. But doing ReLU first will chop off x < 0 first, and we don't want that. 
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        # Don't want to activate it yet! Take into account the action value first. 
        action_value = F.relu(self.action_value(action))  # Not going to calculate batch statistics on this.
        state_action_value = F.relu(torch.add(state_value, action_value))
        # Kinda funky stuff here... I do a double ReLU on the action_value here. 
        # Because ReLU is not a commutative function with add (the order matters), this is kind of sketchy here. But this works well.
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... Saving Checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('... Loading Checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

# %%
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='./ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, name + 'ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003 # This comes from the paper.
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print("Device: {}".format(self.device))
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.tanh(self.mu(x)) # Bound it between -1, 1

        return x

    def save_checkpoint(self):
        print('... Saving Checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... Loading Checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

# %%
class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, 
                 n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, 
                                n_actions=n_actions, name='Actor')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, 
                                n_actions=n_actions, name='TargetActor') # Similar to deep-q network. Off-policy method, same architecture as Actor

        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, 
                                n_actions=n_actions, name='Critic')

        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, 
                                n_actions=n_actions, name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions)) # Mean of 0 over time

        self.update_network_parameters(tau=1) # In Q-network, we don't want to chase a moving target. Update the target network parameters every tau timesteps. 

    def choose_action(self, observation):
        self.actor.eval() # This is very important!!! Tell torch you don't want to calculate the statistics for the batchnorm. Same with dropouts

        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device) # Make sure it's a cuda tensor
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device) # Add a slight noise to the action
        
        self.actor.train()

        return mu_prime.cpu().detach().numpy() # Can't pass a torch tensor in OpenAI gym.

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward      =  torch.tensor(reward, dtype=torch.float).to(self.critic.device)
        done        =  torch.tensor(done, dtype=torch.float).to(self.critic.device)
        new_state   =  torch.tensor(new_state, dtype=torch.float).to(self.critic.device)
        action      =  torch.tensor(action, dtype=torch.float).to(self.critic.device)
        state       =  torch.tensor(state, dtype=torch.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j]) 
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad() # Zero out the grads so that the gradients from previous steps don't interfere with the calculation. 
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval() # Loss calculation of our actor network
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update the parameters for the target actor and target critic
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        ''' tau : parameter that allows the update of target network to gradually approach the evaluation networks. Nice slow convergence. '''
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                       (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                       (1-tau)*actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()





