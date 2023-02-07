# This will be a test program, where a random walker will be confined to a [0,L]
# domain with periodic boundaries, and the walker will learn to avoid the system edges
# using PyTorch actor-critic algorithm

import sys
import time
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
    def __init__(self, input_dims, hidden_dims, n_actions, gamma):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.pi1 = nn.Linear(input_dims, hidden_dims) 	# actor; they call it pi because in literature math pi symbol used to denote a policy
        self.v1 = nn.Linear(input_dims, hidden_dims) 	# critic; also denoted as a V in math literature
        self.pi = nn.Linear(hidden_dims, n_actions)
        self.v = nn.Linear(hidden_dims, 1)

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
        R = v[-1] # grab the last value to obtain Q
        
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

    def __init__(self, L, T_MAX, T_UPDATE, n_actions, input_dims, hidden_dims, lr, gamma):
        
        self.lattice_size = L
        self.simulation_time = T_MAX
        self.update_interval = T_UPDATE
        self.n_of_time_snapshots = int(T_MAX/T_UPDATE)
        self.input_dims = input_dims
        self.walker_position = random.randint(0, L-1) # pick random lattice site
        self.flip_coin = random.randint(0,1)
        self.walker_actor_critic = ActorCritic(input_dims, hidden_dims, n_actions, gamma) # assign walker a neural network
        #self.optimizer = T.optim.Adam(self.walker_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999)) # give network an optimizer
        self.optimizer = T.optim.Adam(self.walker_actor_critic.parameters(), lr=lr) # give network an optimizer
        # record the data
        self.record_pdf = np.zeros(self.lattice_size)
        self.record_current = np.zeros(self.lattice_size)
        self.save_score = []
        self.save_loss = []

    def run(self):
        score = 0
        # main loop
        for t_step in range(self.simulation_time):
            # access the state
            observation = np.zeros(self.input_dims) # here input_dims = 2 * observ_radius + 1 
            for i in range (self.input_dims):
                observation[i] = self.walker_position - (self.input_dims-1)/2 + i
            # choose action; returns a list of size 2, with probabilities to jump to the left and to the right
            action = self.walker_actor_critic.choose_action(observation) # is just a number
            # update the system accordingly
            delta_x, reward = 0, 0
            if self.flip_coin == 0: # to eliminate network bias
                if action == 0: # jump to the right
                    delta_x = 1
                else:           # jump to the left
                    delta_x = -1
            else:
                if action == 0: # jump to the right
                    delta_x = -1
                else:           # jump to the left
                    delta_x = 1
                    
            if t_step > 0.9*self.simulation_time:
                self.record_pdf[self.walker_position] += 1/(0.1*self.simulation_time)
                self.record_current[self.walker_position] += delta_x/(0.1*self.simulation_time)
            # punish the walker if it crosses the periodic boundary
            if self.walker_position + delta_x > self.lattice_size - 1:
                self.walker_position = self.walker_position + delta_x - self.lattice_size
                reward = -100
            if self.walker_position + delta_x < 0:
                self.walker_position = self.walker_position + delta_x + self.lattice_size
                reward = -100
            else: 
                self.walker_position = self.walker_position + delta_x
                reward = 1
                                
            # record observation, action, and corresponding reward
            self.walker_actor_critic.remember(observation, action, reward)
            score += reward

            # update networks
            if t_step % self.update_interval == 0:
                # non-communicating walkers
                loss = self.walker_actor_critic.calc_loss()
                self.optimizer.zero_grad() # reset the gradients of model parameters
                loss.backward() # backpropagate the prediction loss
                self.optimizer.step() # adjust the parameters by the gradients collected in the backward pass
                self.walker_actor_critic.clear_memory() # clear states, actions, rewards arrays 
                
                self.save_score.append(score)
                self.save_loss.append(loss)

        return self.save_score, self.save_loss, self.record_pdf, self.record_current


# our main
if __name__ == '__main__':
    start_time = time.time() # time your simulation
    L = 20          # lattice size
    T_MAX = 100000    # total number of steps
    T_UPDATE = 100  # update networks every this number of steps
    N_RUNS = 100
    lr = 1e-4       # learning rate
    gamma = 1.0
    n_actions = 2   # jump to the left or to the right
    observation_radius = 2
    input_dims = 2*observation_radius + 1 # size of the observation state
    hidden_dims = 10

    n_of_time_snapshots = int(T_MAX/T_UPDATE)
    total_score = np.zeros(n_of_time_snapshots)
    total_loss = np.zeros(n_of_time_snapshots)
    pdf = np.zeros(L)
    current = np.zeros(L)

    dumm_walkers = 0
    for run in range(0,N_RUNS):
        print("Run", run + 1)

        simulation = Environment(L, T_MAX, T_UPDATE, n_actions, input_dims, hidden_dims, lr, gamma)
        score, loss, pdf_per_run, current_per_run = simulation.run()

        # record results of a single run 
        for timestep in range(0,n_of_time_snapshots):
            total_score[timestep] += score[timestep] / N_RUNS
            total_loss[timestep] += loss[timestep] / N_RUNS
            
        # see if the derivative of the end tail of the score curve is positive
        # if yes, then the RW learned to avoid the periodic boundary
        score_growing = False
        delta = 0
        for timestep in range(n_of_time_snapshots - 10, n_of_time_snapshots):
            delta += score[timestep] - score[timestep - 1] # should be positive
        if (delta > 0):
            score_growing = True
        else:
            dumm_walkers += 1
       
        # record pdf and current only if the RW learned to do the thing  
        if(score_growing):
            for x in range(0,L):
                pdf[x] += pdf_per_run[x] 
                current[x] += current_per_run[x] 

    print("Python execution time: %s seconds " % (time.time() - start_time))
    print("Number of walkers that haven't figured it out: ", dumm_walkers)
    
    # normalise pdf and current arrays
    for x in range(0,L):
        pdf[x] /= (N_RUNS - dumm_walkers)
        current[x]  /= (N_RUNS - dumm_walkers)

    # record text output file
    shift = int(L/2)
    x_axis = np.arange(0,L)
    x_axis_shifted = np.arange(-shift,shift)
    pdf_shifted = np.roll(pdf,shift)
    current_shifted = np.roll(current,shift) 
    output_pdf_current = "AC_output_exclude/pdf_and_current_L" + str(L) + "_TMAX" + str(T_MAX) + "_UPD" + str(T_UPDATE) + "_NRUNS" + str(N_RUNS) +"_lr" + str(lr) +  ".txt"
    np.savetxt(output_pdf_current, np.c_[x_axis,pdf,current, pdf_shifted, current_shifted])

    # plot reward vs time
    tot_reward_filename = "AC_output_exclude/total_reward_L" + str(L) + "_TMAX" + str(T_MAX) + "_UPD" + str(T_UPDATE) + "_NRUNS" + str(N_RUNS) +"_lr" + str(lr) +  ".png"
    plt.figure(figsize=(10,8))
    axes = plt.gca()
    plt.xlabel("time", fontsize=24)
    plt.ylabel("Total Reward", fontsize=24)
    plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
        width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
    time = np.arange(n_of_time_snapshots)
    plt.plot(time*T_UPDATE, total_score)
    plt.tight_layout()
    plt.savefig(tot_reward_filename, format="png", dpi=600)
    #plt.show()
    
    # plot loss vs time
    tot_loss_filename = "AC_output_exclude/total_loss_L" + str(L) + "_TMAX" + str(T_MAX) + "_UPD" + str(T_UPDATE) + "_NRUNS" + str(N_RUNS) +"_lr" + str(lr) +  ".png"
    plt.figure(figsize=(10,8))
    axes = plt.gca()
    plt.xlabel("time", fontsize=24)
    plt.ylabel("Total Loss", fontsize=24)
    plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
        width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
    time = np.arange(n_of_time_snapshots)
    plt.plot(time*T_UPDATE, total_loss)
    plt.tight_layout()
    plt.savefig(tot_loss_filename, format="png", dpi=600)
    #plt.show()
    
    # plot pdf
    pdf_filename = "AC_output_exclude/PDF_L" + str(L) + "_TMAX" + str(T_MAX) + "_UPD" + str(T_UPDATE) + "_NRUNS" + str(N_RUNS) +"_lr" + str(lr) +  ".png"
    plt.figure(figsize=(10,8))
    axes = plt.gca()
    plt.xlabel("X", fontsize=24)
    plt.ylabel("PDF", fontsize=24)
    plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
        width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
    plt.plot(pdf, linestyle='--', marker='o', color='k')
    plt.tight_layout()
    plt.savefig(pdf_filename, format="png", dpi=600)
    #plt.show()
    
     # plot pdf shifted
    pdf_shifted_filename = "AC_output_exclude/PDF_shifted_L" + str(L) + "_TMAX" + str(T_MAX) + "_UPD" + str(T_UPDATE) + "_NRUNS" + str(N_RUNS) +"_lr" + str(lr) +  ".png"
    plt.figure(figsize=(10,8))
    axes = plt.gca()
    plt.xlabel("X", fontsize=24)
    plt.ylabel("PDF", fontsize=24)
    plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
        width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
    plt.plot(x_axis_shifted, pdf_shifted, linestyle='--', marker='o', color='k')
    plt.tight_layout()
    plt.savefig(pdf_shifted_filename, format="png", dpi=600)
    #plt.show()
    
    # plot current shifted
    current_filename = "AC_output_exclude/Current_L" + str(L) + "_TMAX" + str(T_MAX) + "_UPD" + str(T_UPDATE) + "_NRUNS" + str(N_RUNS) +"_lr" + str(lr) +  ".png"
    plt.figure(figsize=(10,8))
    axes = plt.gca()
    plt.xlabel("X", fontsize=24)
    plt.ylabel("Current", fontsize=24)
    plt.tick_params(axis='both', which='major', direction = 'in', length = 10 , 
        width = 1, labelsize=24, bottom = True, top = True, left = True, right = True)
    plt.plot(x_axis_shifted,current_shifted, linestyle='--', marker='o', color='k')
    plt.tight_layout()
    plt.savefig(current_filename, format="png", dpi=600)
    #plt.show()