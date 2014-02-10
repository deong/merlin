# merlin/graphs.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merlin, the generator for multitask environments
# for reinforcement learners.
#
# This module defines functionality for generating different state
# transition graphs.
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
import sys
import random
import math
import networkx as nx
import numpy as np
import numpy.random as npr
import pybrain.datasets
import pybrain.tools.shortcuts
import pybrain.supervised.trainers
import matplotlib.pyplot as mpl
import merlin.values as values

#
# create a random graph with uniform out-degree
#
# parameters:
#	nstates: number of states
#	nactions: number of actions
# returns:
#	a directed multigraph with each node having out-degree exactly
#	equal to edges and in-degree > 0.
#
def rand_graph_uniform_degree(nodes, edges):
	"""Create a random graph representing the transition graph for a
	random MDP."""
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
	xs = np.sort(np.asarray([random.random() for i in range(nodes + 1)]))
	diffs = xs[1:] - xs[:nodes]
	diffs = sum(dout) / sum(diffs) * diffs
	din = [int(round(x)) for x in diffs]

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
		node = random.randint(0, nodes - 1)
		if din[node] > 1:
			din[node] -= 1
			total_in -= 1

	# finally, a last sanity check...if we don't have enough inbound
	# edges, add some more. Note that I'm not sure this ever happens,
	# but it's easy enough to handle.
	while total_in < sum(dout):
		node = random.randint(0, nodes - 1)
		din[node] += 1
		total_in += 1

	# if we did this right, the sums should be guaranteed to match
	assert(sum(din) == sum(dout))

	# generate a random directed multigraph with the specified degree
	# sequences
	tgraph = nx.directed_configuration_model(din, dout)

	# now label each node with a number so we can refer to it later
	# for i, node in enumerate(tgraph):
	# 	tgraph.node[i]['node_num'] = i
	return tgraph



#
# take an arbitrary directed graph and make it strongly connected,
# maintaining the total number of incoming and outgoing edges for
# each vertex.
#
# parameters:
#   G: the input graph to check
# returns:
#   G with edges added if needed to make the graph connected
#   
def make_strongly_connected(G):
	components = nx.strongly_connected_components(G)
	num_components = len(components)
	if num_components == 1:
		return G

	# for each connected component, connect one node to a node in
	# the successor component, and delete an edge to make up for it.
	# which edge to delete isn't trivial -- it only needs to be an edge
	# that is somehow redundant in terms of connecting the graph. Our
	# approach is to delete an edge at random, and keep trying until
	# either the graph is connected or we exhaust the number of tries.
	attempts = 0
	max_attempts = num_components * math.log(num_components,2)
	while num_components > 1 and attempts < max_attempts:
		for index in range(num_components):
			source_comp = components[index]
			target_comp = components[(index+1) % num_components]

			# pick a random vertex from the source component and connect it
			# to a vertex in the target component, deleting one of the outgoing
			# edges from the source vertex to keep the degree constant
			source_vertex = source_comp[npr.randint(len(source_comp))]
			target_vertex = target_comp[npr.randint(len(target_comp))]
			source_edges = list(G[source_vertex].keys())
			G.remove_edge(source_vertex, source_edges[npr.randint(len(source_edges))])
			G.add_edge(source_vertex, target_vertex)
		components = nx.strongly_connected_components(G)
		num_components = len(components)
		attempts += 1
	return G

			
# construct a continuous MDP based on a random graph
#
# The basic idea is to construct a random graph with the desired properties,
# then fit each node and edge with a real number and train a neural network to
# predict the new state value given a state/action pair.
#
# parameters:
#   G: state transition graph
#   R: reward structure
#   indim: number of state variables per state
#   hidden_units: number of hidden units in the network
#   training_log: the name of a file to write predicted dynamics to
#   
def make_continuous_mdp(G, R, inpd, hidden_units):
	# build up a training set for the neural network; input is each component
	# of the current state vector + one action, and output is the components
	# of the successive state vector
	training_set = pybrain.datasets.SupervisedDataSet(inpd + 1, inpd)

	# first, assign a random vector to each state/action pair in the graph
	# using the fractal landscape method
	num_points = len(G)
	state_values = []
	for d in range(inpd):
		# vals = make_fractal(num_points, 0.75)
		vals = values.make_gaussian_walk(num_points, 1.0)
		state_values.append(vals)
	state_values = list(zip(*state_values))
	
	# create a map from node to state vector	  
	state_value_map = {}
	for i, node in enumerate(G):
	 	state_value_map[node] = state_values[i]

	# create a list of actions for each node
	action_value_map = {}
	for node in G:
		num_actions = len(G.successors(node))
		step = 2.0 / num_actions
		min_val = -1.0
		max_val = min_val + step
		for index, succ in enumerate(G.successors(node)):
			# FIXME: customized for two actions
			# action_value_map[(node, succ)] = npr.random() * 2.0 - 1.0
			aval = npr.random() * (max_val - min_val) + min_val
			action_value_map[(node, succ)] = aval
			min_val += step
			max_val += step

	# now go back through the graph, for each node connecting it via an action
	# to a successor node
	for node in G:
		for index, succ in enumerate(G.successors(node)):
			inp  = np.append(state_value_map[node], action_value_map[(node, succ)])
			outp = state_value_map[succ]
			training_set.addSample(inp, outp)

	# finally, create a train a network
	# hidden units:
	#    1 * inpd = very fast, not much improvement
	#    2 * inpd = 5+ minutes, some improvement
	#    3 * inpd = 6 minutes, some improvement
	#    4 * inpd = 7.5 seconds
	nnet = pybrain.tools.shortcuts.buildNetwork(inpd + 1, hidden_units, inpd, bias=True)
	trainer = pybrain.supervised.trainers.BackpropTrainer(nnet, training_set)
	print('Training neural network on state dynamics...this may take a while...', file=sys.stderr)
	errors = trainer.trainEpochs(2000)
	#errors = trainer.trainUntilConvergence(maxEpochs=2000)

	return (nnet, training_set, state_value_map, action_value_map)



# randomly change fuzz_factor percent of the weights of the given network
#
# parameters:
#   net: the trained network to fuzz
#   frac: the percentage of weights to randomly change
#   scale: the amount to change the selected weights by
#   
def fuzz_neural_net(net, frac, scale):
	weights = net.params
	fweights = [x if npr.random() >= frac else npr.normal(x, scale) for x in weights]
	net._setParameters(fweights)
	return net
	


