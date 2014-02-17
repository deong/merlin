# merlin/rewards.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merlin, the generator for multitask environments
# for reinforcement learners.
#
# This module defines functions for generating reward distributions.
# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
import numpy as np
import numpy.random as npr
import scipy.linalg as sla
import pybrain.datasets
import pybrain.tools.shortcuts
import pybrain.supervised.trainers

#
# create random reward structure for an (nstates, nactions) MDP
#
# parameters:
#	nstates:  number of states
#	nactions: number of actions
#	covmat:	  nxn covariance matrix, where n=number of tasks
# returns:
#	an (nstates x nactions x ntasks) reward matrix
#
def mvnrewards(nstates, nactions, mu, covmat):
	"""Create a random reward structure for an (nstates, nactions) MDP
	where the rewards for each pair of tasks are correlated according
	to the specified covariance matrix."""
	# make sure covmat is positive definite; raise an exception if it
	# isn't. Note that the multivariate_normal call succeeds either way
	# but the results aren't physically meaningful if the matrix isn't
	# semi-positive definite, and we'd rather bail than generate data
	# that doesn't match what the user asked for.
	sla.cholesky(covmat)
	ntasks = covmat.shape[0]
	rewards = npr.multivariate_normal(mu, covmat, (nstates, nactions))
	return rewards



#
# convert a correlation matrix to a covariance matrix with given standard deviations
#
# parameters:
#   R: the given corelation matrix
#   sigma: vector of standard deviations for each dimension
# returns:
#   a covariance matrix matching the input specifications
#   
def cor2cov(R, sigma):
	return np.diag(sigma).dot(R).dot(np.diag(sigma))



#
# check if a given matrix is positive definite
#
# parameters:
#   A: an nxn matrix
# returns:
#   True if A is positive definite, false otherwise
#
def is_pos_def(A):
	try:
		sla.cholesky(A)
	except sla.LinAlgError:
		return False
	return True

#
# learn an approximation model for the given reward function
#
# parameters:
#   G: the state transition graph
#   rewards: an nxmxk matrix of state/action reward values
#   max_epochs: maximum number of training epochs for the networks
# returns:
#   a trained neural network 
#
def learn_reward_function(G, hidden_units, max_epochs):
	# grab any node and use it to find the dimensionality of the state space
	dim = len(G.node[0]['state'])
	
	# grab any edge and use it to find the dimensionality of the reward space
	(src, dest, key) = G.edges(keys=True)[0]
	num_tasks = len(G.edge[src][dest][key]['reward'])
	
	training_set = pybrain.datasets.SupervisedDataSet(dim + 1, num_tasks)
	
	# for each node in the graph, map the state + action onto a reward
	for state_index, node in enumerate(G):
		s = G.node[node]['state']
		for action_index, (_, succ, key) in enumerate(G.out_edges(node, keys=True)):
			a = G.edge[node][succ][key]['action']
			training_set.addSample(np.append(s, a), G.edge[node][succ][key]['reward'])
	
	# finally, create a train a network
	nnet = pybrain.tools.shortcuts.buildNetwork(dim + 1, hidden_units, num_tasks, bias=True)
	trainer = pybrain.supervised.trainers.BackpropTrainer(nnet, training_set)
	errors = trainer.trainUntilConvergence(maxEpochs=max_epochs)
	
	return (nnet, training_set)
	

# write the reward information directly into the graph as an annotation on the edges
#
# parameters:
#   G: the transition graph
#   rewards: an nxmxk matrix of rewards, one for each state-action pair
#   
def annotate_rewards(G, rewards):
	for i, node in enumerate(G):
		for j, (_, succ, key) in enumerate(G.out_edges(node, keys=True)):
			G.edge[node][succ][key]['reward'] = rewards[i,j]
