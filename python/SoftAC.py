# This will be a test program, where a random walker will be confined to a [0,L]
# domain with periodic boundaries, and the walker will learn to avoid the system edges
# using PyTorch actor-critic algorithm

import os
import argparse
import numpy as np
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# simulation parameters
parser = argparse.ArgumentParser(description = 'RW, where the walker learns to avoid the system boundaries')
parser.add_argument('')
parser.add_argument('--size', type=int, default=10, metavar='L', help='size of the lattice')
parser.add_argument('--epoches', type=int, default=10, metavar='E', help='number of epochs')
parser.add_argument('--period', type=int, default=100, metavar='T', help='number of timesteps per epoch')
parser.add_argument('--radius', type=int, default=3, metavar='R', help='observation radius')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=123, metavar='seed', help='random seed (default: 123)')
args = parser.parse_args()

SavedAction = namedtuple('SavedAction', ['action', 'value'])
torch.manual_seed(args.seed)

# environment class -- our periodic 1d lattice with a walker 
class Environment():

	def __init__(self):
		self.size = args.size					# lattice size
		self.radius = args.radius  				# observation radius
		self.position = random.randint(size-1)	# initial position of the walker, chosed randomly
		self.record_positions = []				# keep track of the walker stepping history
		record_positions.append(position)
		
	def observation(self):
		# record spatial cordinates of the walker surroundings [X - radius, ..., X, ..., X + radius] 
		State = numpy.zeros(5, dtype=int)
		for i in range (2 * radius + 1):
			State[i] = position - radius + i
		return State

	def update_position(self, delta_x, reward):
		# punish the walker if it crosses the periodic boundary
		if position + delta_x > size - 1:
			position = position + delta_x - size + 1
			reward = reward - 100
		if position + delta_x < 0:
			position = position + delta_x + size - 1
			reward = reward - 100
		else: 
			position = position + delta_x
			reward = reward + 1
		
		record_positions.append(position)
		return reward


env = Environment() # initialize an instance of the Environment class -- our environment where the stuff is actually hapenning


# Actor & Critic implementation
class Policy(nn.Module):

	def __init__(self):		
		super(Policy, self).__init__()
		# observation
		self.State = nn.Linear(radius, 5) 	# the agent's observation of the state, which acts as an input
		# actor's layer 
		self.action_head = nn.Linear(5,2)	# action will provide us with an output probability to jump to the right 
		# critic's layer
		self.value_head = nn.Linear(5,1)	# criticl will provide us with his own judgement of the action that will be taken

		# action & reward buffer
		self.saved_actions = []
		self.rewards = []

	# don't need to call it directly; by calling Policy(input), will be executed automatically
	def forward(self, x):  
		x = F.relu(self.State(x))
		
		# actor: choses action to take from 'State' by returning probability to jump to the right
		action_prob = F.softmax(self.action_head(x), dim=-1)

		# critic: evaluates being in 'State'
		state_value = self.value_head(x)
		
		# return probability and critic's evaluation
		return action_prob, state_value


model = Policy() # initialize an instance of the Policy class -- our neural network, which does the reinforcement learning 
optimizer = optim.Adam(model.parameters(), lr=3e-2) # adjusts model parameters to reduce model error in each training step 
eps = np.finfo(np.float32).eps.item() # machine epsilon, add to avoid infinities


# selects action, based on Policy, and records action and critic's judgement
def select_action(state):
	# feed the observation to a NN
	state = torch.from_numpy(state).float()
	probs, critic_estimate = model(state)
	
	# create a categorical distribution over the list of probabilities of actions
	m = Categorical(probs)

	# and sample an action using the distribution
	action = m.sample()
	
	model.saved_actions.append(SavedAction(m.log_prob(action), critic_estimate))

	return action.item()


# training code, calculates actor & critic loss and performs backpropagation
def finish_episode():
	
	R = 0 # reward 
	saved_actions = model.saved_actions
	policy_losses = [] 	# list to save actor (policy) loss
	value_losses = []	# list to save critic (value) loss
	returns = []		# list to save true values (the agent's goal is to maximize returns)

	# calculate the true value using rewards returned from the environment
	for r in model.rewards[::-1]: # [<start>:<stop>:<step>] --> [::-1] means it starts from the end towards the first element
		# calculate the descounted value
		R = r + args.gamma * R; # gamma is a discount factor, i.e., memory? 
		returns.insert(0, R) # insert R value always at the beginning of the list
		
		
	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps) # why do you do that? some sort of normalization?
	
	for (action, value), R in zip(saved_actions, returns):
		advantage = R - value.item()
		
		# calculate actor (policy) loss
		policy_losses.append(-log_prob * advantage)
		
		# calculate critic (value) loss using L1 smooth loss
		value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]))) # computes |value - R| difference in a specified way
	
	# reset gradients
	optimizer.zero_grad()
	
	# sum up all the values of policy_losses and value_losses
	loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
	
	# perform backdrop
	loss.backward()
	optimizer.step()
	
	# reset rewards and action buffer
	del model.rewards[:]
	del model.saved_actions[:]
	

# main loop




	
	
	dice = random.uniform(0,1)
	
	if dice < prob:
		action = 1
	else: 
		action = - 1



# output

