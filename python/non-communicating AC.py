# This will be a test program, where a random walker will be confined to a [0,L]
# domain with periodic boundaries, and the walker will learn to avoid the system edges
# using PyTorch actor-critic algorithm

import sys
import numpy as np
import random 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt 

# Asyncronous Advantage Actor Crtic algorithm (A3C)
# Taken from: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.pi1 = nn.Linear(input_dims, 10) 	# actor; they call it pi because in literature math pi symbol used to denote a policy
        self.v1 = nn.Linear(input_dims, 10) 	# critic; also denoted as a V in math literature
        self.pi = nn.Linear(10, n_actions)
        self.v = nn.Linear(10, 1)

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
        pi, _ = self.forward(state) # produces sets of logits
        probs = T.softmax(pi, dim=1) # converts logits to probabilities (torch object)
        dist = Categorical(probs) # feeds torch object to generate a list of probs (numpy object ?)
        action = dist.sample().numpy()[0] # sample list of probs and return the action

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
class Environment():

    def __init__(self, N, L, T_MAX, T_UPDATE, n_actions, input_dims, lr, gamma):
        
        self.n_of_walkers = N
        self.lattice_size = L
        self.simulation_time = T_MAX
        self.update_interval = T_UPDATE
        self.n_of_time_snapshots = int(T_MAX/T_UPDATE)
        self.input_dims = input_dims
        self.walkers_positions = np.zeros(N)
        self.walkers_positions = [random.randint(0, L-1) for i in range(N)] # distribute N walkers over a lattice of size L at random
        #print("Initial conditions")
        #print(self.walkers_positions)
        self.walkers_local_actor_critic = [ActorCritic(input_dims, n_actions, gamma) for i in range(N)] # assign to each walker a neural network
        # create a list of optimizers
        self.optimizer = []
        for walker in range(N):
            optim = T.optim.Adam(self.walkers_local_actor_critic[walker].parameters(), lr=lr, betas=(0.92, 0.999))
            self.optimizer.append(optim)
        # record the data
        self.record_positions = np.zeros(shape=(self.n_of_time_snapshots, N))
        self.save_score = []

    def run(self):
        score = 0
        # main loop
        for t_step in range(self.simulation_time):
            for walker in range(self.n_of_walkers):
                # access the state
                observation = np.zeros(self.input_dims) # here input_dims = 2 * observ_radius + 1 
                for i in range (self.input_dims):
                    observation[i] = self.walkers_positions[walker] - (self.input_dims-1)/2 + i
                # choose action; returns a list of size 2, with probabilities to jump to the left and to the right
                action = self.walkers_local_actor_critic[walker].choose_action(observation) # is just a number
                # update the system accordingly
                delta_x, reward = 0, 0
                if action == 0: # jump to the right
                    delta_x = 1
                else:           # jump to the left
                    delta_x = -1
                # punish the walker if it crosses the periodic boundary
                if self.walkers_positions[walker] + delta_x > self.lattice_size - 1:
                    self.walkers_positions[walker] = self.walkers_positions[walker] + delta_x - self.lattice_size + 1
                    reward = -100
                if self.walkers_positions[walker] + delta_x < 0:
                    self.walkers_positions[walker] = self.walkers_positions[walker] + delta_x + self.lattice_size - 1
                    reward = -100
                else: 
                    self.walkers_positions[walker] = self.walkers_positions[walker] + delta_x
                    reward = 1
                                 
                # record observation, action, and corresponding reward
                self.walkers_local_actor_critic[walker].remember(observation, action, reward)
                score += reward

            # update networks
            if t_step % self.update_interval == 0:
                """"
                total_loss = 0
                # compute loss for all walkers
                for walker in range(self.n_of_walkers):
                    compute_loss = self.walkers_local_actor_critic[walker].calc_loss()
                    #print(compute_loss)
                    total_loss += compute_loss.item()
                # average gradients and backpropagate the average
                av_loss = total_loss / self.n_of_walkers
                print("Average loss", av_loss)
                for walker in range(self.n_of_walkers):
                    loss = T.tensor(av_loss, requires_grad=True)
                    self.optimizer[walker].zero_grad() # reset the gradients of model parameters
                    loss.backward() # backpropagate the prediction loss
                    self.optimizer[walker].step() # adjust the parameters by the gradients collected in the backward pass
                """
                # non-communicating walkers
                for walker in range(self.n_of_walkers):
                    loss = self.walkers_local_actor_critic[walker].calc_loss()
                    self.optimizer[walker].zero_grad() # reset the gradients of model parameters
                    loss.backward() # backpropagate the prediction loss
                    self.optimizer[walker].step() # adjust the parameters by the gradients collected in the backward pass

                # clear states, actions, rewards arrays
                [self.walkers_local_actor_critic[walker].clear_memory() for walker in range(self.n_of_walkers)]     
                
                self.save_score.append(score)
                time_index = int(t_step/self.update_interval)
                for walker in range(self.n_of_walkers):
                    self.record_positions[time_index][walker] = self.walkers_positions[walker]
                
                #print(self.record_positions)
                print('timestep', t_step, 'reward %.1f' % score)
        
        #print(self.record_positions)

        return self.save_score, self.record_positions


# our main
if __name__ == '__main__':

    N = 100          # number of particles
    L = 20          # lattice size
    T_MAX = 4000    # total number of steps
    T_UPDATE = 10  # update networks every this number of steps
    N_RUNS = 1000
    lr = 1e-4       # learning rate
    gamma = 1.0
    n_actions = 2   # jump to the left or to the right
    observation_radius = 2
    input_dims = 2*observation_radius + 1 # size of the observation state

    n_of_time_snapshots = int(T_MAX/T_UPDATE)
    total_score = np.zeros(n_of_time_snapshots)
    pdf = np.zeros(shape=(n_of_time_snapshots,L))

    for run in range(0,N_RUNS):
        print("Run", run + 1)
        #seed = random.randint(0, sys.maxsize)
        #T.manual_seed(seed)

        simulation = Environment(N, L, T_MAX, T_UPDATE, n_actions, input_dims, lr, gamma)
        score, positions = simulation.run()
        #print(positions)

        # record results of a single run into pdf 
        for timestep in range(0,n_of_time_snapshots):
            total_score[timestep] += score[timestep] / N_RUNS
            # get the pdf too
            for i in range(0, N):
                _index = int(positions[timestep][i])
                pdf[timestep][_index] += 1.0 / N
    
    # normalize pdf
    for timestep in range(0,n_of_time_snapshots):
        for i in range(0, L):
            pdf[timestep][i] /= N_RUNS
    #print("normalized PDF")
    #print(pdf)

    # plot the reward vs time
    plt.figure(figsize=(10,8))
    axes = plt.gca()
    plt.xlabel("time", fontsize=24)
    plt.ylabel("Total Reward", fontsize=24)
    plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
        width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
    time = np.arange(n_of_time_snapshots)
    plt.plot(time*T_UPDATE, total_score)
    plt.tight_layout()
    plt.savefig("reward.png", format="png", dpi=600)
    #plt.show()
    
    # plot space-time probability distribution and make a movie
    import imageio
    filenames = []
    for timestep in range(0,n_of_time_snapshots):
        plt.figure(figsize=(10,8))
        plt.tight_layout()
        axes = plt.gca()
        axes.set_xlim([0, L-1])
        axes.set_ylim([0, 0.15])
        plt.title("t = " + str(timestep*T_UPDATE), fontsize = 24)
        plt.xlabel("X", fontsize=24)
        plt.ylabel("PDF(X,t)", fontsize=24)
        plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
            width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
        X = np.arange(L)
        plt.scatter(X, pdf[timestep], c="r")
        plt.plot(X, pdf[timestep], c='k')
        plt.savefig("pdf/t" + str(timestep*T_UPDATE) + ".png", format="png", dpi=600)
        filenames.append("pdf/t" + str(timestep*T_UPDATE) + ".png")
    # generates gif
    with imageio.get_writer('movie.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
