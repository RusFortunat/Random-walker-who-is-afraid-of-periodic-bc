# This will be a test program, where a random walker will be confined to a [0,L]
# domain with periodic boundaries, and the walker will learn to avoid the system edges
# using PyTorch actor-critic algorithm

import os
import numpy as np
import random 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

T.manual_seed(123)
foo = random.SystemRandom()

# Creates unified data table for all local_actor_critic instances
class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        # read about this carefully later!!!
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

# Asyncronous Advantage Actor Crtic algorithm (A3C)
# Taken from: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.pi1 = nn.Linear(input_dims, 128) 	# actor; they call it pi because in literature math pi symbol used to denote a policy
        self.v1 = nn.Linear(input_dims, 128) 	# critic; also denoted as a V in math literature
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

        self.rewards = []
        self.actions = []
        self.states = []

    def forward(self, state):
        # input state fed to hidden network
        pi1 = F.relu(self.pi1(state)) 
        v1 = F.relu(self.v1(state))
        # from hidden network to output
        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v

    # simply choose the action from the observation state
    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        #print("State", state)
        pi, _ = self.forward(state) # produces sets of logits
        #print("Pi", pi)
        probs = T.softmax(pi, dim=1) # converts logits to probabilities
        #print("probs", probs)
        dist = Categorical(probs) # feeds torch object to generate a list of probs
        #print("dist", dist)
        action = dist.sample().numpy()[0] # sample list of probs and return the action
        #print("Action object", action)

        return action

	# calculate the cummulative reward, which is based only on critic observations 'v'
    def calc_R(self):
        states = T.tensor(self.states, dtype=T.float)
        # standalone _ here is an unimportant variable, which we put because one have to submit a pair of variables to 'forward'
        _, v = self.forward(states) # get the list of v's for the list of 
        R = v[-1] # grab the last value

        # calculate reward here
        batch_return = []
        for reward in self.rewards[::-1]: # start from the last element of the array 
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    # calculate total loss
    def calc_loss(self):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R() # returns calculated

        pi, values = self.forward(states) # get the actions and values
        values = values.squeeze() # removes axes of the length one
        critic_loss = (returns-values)**2 # the actual reward that 

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

     # record states, actions, and rewards
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []


# this is where the simulation will be hapenning
class Agent(mp.Process):
    def __init__(self, N_GAMES, T_MAX, T_UPDATE, size, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, name, global_ep_idx):
        super(Agent, self).__init__()
        self.size = size
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma) # a network for a single random walker
        self.global_actor_critic = global_actor_critic  # contains the parameters for all RW networks
        self.name = 'w%02i' % name # walker's name in the output table
        self.episode_idx = global_ep_idx
        self.optimizer = optimizer
        self.record_positions = []
        self.N_GAMES, self.T_MAX, self.T_UPDATE = N_GAMES, T_MAX, T_UPDATE
        self.input_dims = input_dims

    def run(self):
        t_step = 1
        position = foo.randint(0,self.size-1) # place a walker at random
        self.record_positions.append(position)

        while self.episode_idx.value < self.N_GAMES:
            score = 0
            self.local_actor_critic.clear_memory() # clear states, actions, rewards arrays

            while t_step < self.T_MAX:
                # record the state
                observation = np.zeros(self.input_dims) # here input_dims = 2 * observ_radius + 1 
                for i in range (self.input_dims):
                    observation[i] = position - (self.input_dims-1)/2 + i
                # choose action; returns a list of size 2, with probabilities to jump to the left and to the right
                action = self.local_actor_critic.choose_action(observation) # is just a number
                # update the system accordingly
                delta_x, reward = 0, 0
                if action == 0: # jump to the right
                     delta_x = 1
                else:           # jump to the left
                    delta_x = -1
                # punish the walker if it crosses the periodic boundary
                if position + delta_x > self.size - 1:
                    position = position + delta_x - self.size + 1
                    reward = reward - 100
                if position + delta_x < 0:
                    position = position + delta_x + self.size - 1
                    reward = reward - 100
                else: 
                    position = position + delta_x
                    reward = reward + 1
                # record new positions
                self.record_positions.append(position) 
                # record observation, action, and corresponding reward
                self.local_actor_critic.remember(observation, action, reward)
                
                # update the network
                if t_step % self.T_UPDATE == 0:
                    loss = self.local_actor_critic.calc_loss() # calculate the total A3C loss
                    self.optimizer.zero_grad() # reset the gradients of model parameters
                    loss.backward() # backpropagate the prediction loss
                    # pass parameters to global Adam optimizer
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        # why is this done ? to collect all local_param and get from them global_param?
                        global_param._grad = local_param.grad # .grad is not writable, this is why we use ._grad
                    # adjust the parameters by the gradients collected in the backward pass
                    self.optimizer.step() # global_param is being optimized, as it seems
                    # load model weights from global_actor_critic
                    self.local_actor_critic.load_state_dict(
                            self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory() # clear states, actions, rewards arrays

                score += reward
                t_step += 1
                
            with self.episode_idx.get_lock(): # multiprocessing object, which doesn't need to be concerned about the source of the Lock
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)

# our main
if __name__ == '__main__':
    size = 100
    lr = 1e-4
    n_actions = 2
    observation_radius = 2
    input_dims = 2*observation_radius + 1
    N_GAMES = 10 # number of independent runs
    T_MAX = 1000
    T_UPDATE = 50
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)
    # size, global_actor_critic, optimizer, input_dims, n_actions, gamma, name, global_ep_idx
    workers = [Agent(N_GAMES,
                    T_MAX,
                    T_UPDATE,
                    size,
                    global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    name=i,
                    global_ep_idx=global_ep) for i in range(mp.cpu_count() - 1)]
    [w.start() for w in workers]
    [w.join() for w in workers]