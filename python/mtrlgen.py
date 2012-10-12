#!/usr/bin/env python
#
# Problem generator for multi-task reinforcement learning problems
#
# Usage:
#
#
# License:
# 
# Copyright 2012 Deon Garrett <deong@acm.org>
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#	  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
import numpy as np
import scipy as sp
import scipy.linalg as sla
import scipy.stats as sst
import numpy.random as npr
import networkx as nx

#
# create random reward structure for an (nstates, nactions) MDP
#
# parameters:
#	nstates:	 number of states
#	nactions: number of actions
#	covmat:	 nxn covariance matrix, where n=number of tasks
# returns:
#	an (nstates x nactions x ntasks) reward matrix
#	
def randinst1(nstates, nactions, covmat):
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
	mu = [0.0] * ntasks
	rewards = npr.multivariate_normal(mu, covmat, (nstates, nactions))
	return rewards


#
# create a uniform random MDP state transition graph
#
# parameters:
#   nstates: number of states
#   nactions: number of actions
# returns:
#   a directed multigraph with each node having out-degree exactly
#   equal to edges and in-degree > 0.
#
def randgraph1(nodes, edges):
	# need din/dout -- degree sequences
	# dout is easy...every node gets a constant number of outbound edges
	# equal to the number actions in the MDP. din is tougher...we want
	# the in-degree of each node to be random, subject to the constraints
	# that every node must be reachable by at least one edge and the total
	# in-degree over the entire graph must equal the total out-degree.
	dout = [edges] * nodes

	# to compute din, we generate a random sequence of N+1 random numbers,
	# take the difference between adjacent pairs, and scale up to the
	# desired target sum (with rounding)
	xs = np.sort(np.asarray([random.random() for i in range(nodes+1)]))
	diffs = xs[1:] - xs[:nodes]
	diffs = sum(dout) / sum(diffs) * diffs
	din = map(lambda(x): int(round(x)), diffs)

	# at this point, din contains random fan-ins for each node, but we
	# may have nodes with 0 edges, and due to rounding, we may be off
	# by one in the sum as well. So walk the list, bumping any zero values
	# up to one, and then randomly remove any excess we have by decrementing
	# some of the nodes with larger fan-ins
	total_in = sum(din)
	for index, degree in enumerate(din):
		if degree < 1:
			din[index] = 1
			total_in += 1
	# now remove edges randomly until the degrees match
	while total_in > sum(dout):
		node = random.randint(0, nodes)
		if din[node] > 1:
			din[node] -= 1
			total_in -= 1
	# finally, a last sanity check...if we don't have enough inbound
	# edges, add some more. Note that I'm not sure this ever happens,
	# but it's easy enough to handle.
	while total_in > sum(dout):
		node = random.randint(0, nodes)
		din[node] += 1
		total_in += 1
		
	# if we did this right, the sums should be guaranteed to match
	assert(sum(din) == sum(dout))

	# generate a random directed multigraph with the specified degree
	# sequences
	tgraph = nx.directed_configuration_model(din, dout)
	return tgraph



# test the correlations of a generated instance
#
# parameters:
#	D: the generated problem instance
#
# returns:
#	the pearson correlation coefficient between each pair of tasks
#
# The correlations should be between tasks. D is NxMxk, where N is
# the number of states, M the number of actions, and k the number of
# tasks. Thus, we want to extract each NxM submatrix, unroll it into
# a column vector, and compare all k such column vectors for their
# correlation coefficients.
def correlation(D):
	n, m, k = D.shape
	corr = np.zeros([k, k])
	for i in range(0, k):
		for j in range(0, k):
			x = np.reshape(D[:,:,i], n*m)
			y = np.reshape(D[:,:,j], n*m)
			corr[i, j] = (sst.pearsonr(x,y)[0])
	return corr
