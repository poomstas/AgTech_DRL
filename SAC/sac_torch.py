import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent:
    def __init__(self, env, input_dims, n_actions, TB_name, alpha=0.0003, beta=0.0003,
                 gamma=0.99, max_size=1000000, tau=0.005, batch_size=256,
                 layer1_size=256, layer2_size=256, reward_scale=2, cuda_index=0):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions, TB_name=TB_name,
                        name='actor', max_action=env.action_space.high, 
                        fc1_dims=layer1_size, fc2_dims=layer2_size, cuda_index=cuda_index)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, TB_name=TB_name,
                        name='critic_1', fc1_dims=layer1_size, fc2_dims=layer2_size, cuda_index=cuda_index)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, TB_name=TB_name, 
                        name='critic_2', fc1_dims=layer1_size, fc2_dims=layer2_size, cuda_index=cuda_index)
        self.value = ValueNetwork(beta, input_dims, TB_name=TB_name, name='value', cuda_index=cuda_index)
        self.target_value = ValueNetwork(beta, input_dims, TB_name=TB_name, name='target_value', cuda_index=cuda_index)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                (1-tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('... Saving Models ...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
    
    def load_models(self, model_refname, model_path, gpu_indx=0):
        fullpath_actor          = os.path.join(model_path, model_refname + '_actor')
        fullpath_critic_1       = os.path.join(model_path, model_refname + '_critic_1')
        fullpath_critic_2       = os.path.join(model_path, model_refname + '_critic_2')
        fullpath_value          = os.path.join(model_path, model_refname + '_value')
        fullpath_target_value   = os.path.join(model_path, model_refname + '_target_value')

        self.actor.load_checkpoint(fullpath_actor, gpu_indx=gpu_indx)
        self.critic_1.load_checkpoint(fullpath_critic_1, gpu_indx=gpu_indx)
        self.critic_2.load_checkpoint(fullpath_critic_2, gpu_indx=gpu_indx)
        self.value.load_checkpoint(fullpath_value, gpu_indx=gpu_indx)
        self.target_value.load_checkpoint(fullpath_target_value, gpu_indx=gpu_indx)
    
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = self.scale * reward + self.gamma * value_

        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
