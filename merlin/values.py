# merlin/values.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merlin, the generator for multitask environments
# for reinforcement learners.
#
# This module defines functions for assigning state and action values
# to nodes in the transition graph.
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


# annotate the given graph with state values for each node
#
# parameters:
#   G: the state transition graph
#   inpd: the desired dimension of the state space
#   func_type: which type of landscape to generate {'fractal' or 'gaussian')
#   ruggedness: parameter governing how  chaotic the changes are
#   
def annotate_states(G, inpd, func_type, ruggedness):
	if func_type == 'fractal':
		func = make_fractal
	elif func_type == 'gaussian':
		func = make_gaussian_walk
	else:
		raise RuntimeError('illegal state-value function type "{}" given'.format(func_type))

	num_points = len(G)
	state_values = []
	for d in range(inpd):
		vals = func(num_points, ruggedness)
		state_values.append(vals)
	state_values = list(zip(*state_values))

	for index, node in enumerate(G):
		G.node[node]['state'] = state_values[index]




# annotate the graph with state values for each node by taking random walks
#
# unlike the preceding method, this one applies the state values in the
# in which the nodes would be visited by walking through the graphs
def annotate_states_walk(G, inpd, func_type, ruggedness):
	if func_type == 'fractal':
		func = make_fractal
	elif func_type == 'gaussian':
		func = make_gaussian_walk
	else:
		raise RuntimeError('illegal state-value function type "{}" given'.format(func_type))

	start = G.nodes()[0]
	paths = get_all_terminal_paths(G, start)
	for path in paths:
		num_points = len(path)
		state_values = []
		for d in range(inpd):
			vals = func(num_points, ruggedness)
			state_values.append(vals)
		state_values = list(zip(*state_values))

		for index, node in enumerate(path):
			G.node[node]['state'] = state_values[index]




# get_all_terminal_paths
#
# return a list of walks through the graph such that all nodes are visited exactly once
#
# parameters:
#   G: the graph to visit
#   s: the start node
def get_all_terminal_paths(G, s):
	nodes = G.nodes()
	unvisited = nodes[:]
	paths = []
	while unvisited:
		if not s in unvisited:
			s = npr.choice(unvisited)
		path = get_terminal_path(G, s, unvisited)
		paths.append(path)
	return paths
		

# get_terminal_path
#
# create a path through the graph from a start node until no further unvisited nodes
# can be reached
#
# parameters:
#   G: a graph
#   s: a start node
#   unvisited: which nodes have not been visited already
def get_terminal_path(G, s, unvisited):
	path = [s]
	unvisited.remove(s)
	done = False
	while not done:
		done = True
		edges = G.out_edges(s)
		npr.shuffle(edges)
		for (_, dst) in edges:
			if dst in unvisited:
				done = False
				path.append(dst)
				unvisited.remove(dst)
				s = dst
				break
	return path

			
# map a real-valued action value onto each edge in the given transition graph
#
# tries to evenly distribute the action values in the allowed range so that you
# don't end up with two states combining with very similar action values mapping
# onto two very different resulting states in the transition dynamics
#
# parameters:
#   G: the state transition graph
#   action_range: a tuple of (min, max) for action values
#   
def annotate_actions(G, action_range):
	for node in G:
		num_actions = len(G.out_edges(node))
		step = (action_range[1] - action_range[0])/float(num_actions)
		min_val = action_range[0]
		max_val = min_val + step
		for index, (_, succ, key) in enumerate(G.out_edges(node, keys=True)):
			aval = npr.random() * (max_val - min_val) + min_val
			G.edge[node][succ][key]['action'] = aval
			min_val += step
			max_val += step
	

# walk the graph and generate all state/action pairs
#
# parameters:
#   G: the annotated state-transition graph
# returns:
#   a list of state-action tuples
def gen_state_action_pairs(G):
	p = []
	for node in G:
		for (_, succ, key) in G.out_edges(node, keys=True):
			p.append((np.asarray(G.node[node]['state']), G.edge[node][succ][key]['action']))
	return p




# return a 1-D fractal landscape generated using the midpoint displacement method
#
# parameters:
#   n: the number of points in the landscape
#   ruggedness: a parameter determining ruggedness [0.0=very rugged, 1.0=very smooth]
#   rng: initial range of the random number generator
#   seed: baseline value of the landscape
# returns:
#   a list of points mapping out landscape heights
#   
def make_fractal(n, ruggedness, rng=1.0, seed=0):
	points = [npr.random(), npr.random()]
	return make_fractal_aux(n, points, ruggedness, rng)

# helper function for the fractal generation
def make_fractal_aux(n, points, ruggedness, rng):
	if len(points) >= n:
		return points[:n]
	newpoints = [points[0]]
	for i in range(len(points)-1):
		p1 = points[i]
		p2 = points[i+1]
		newval = (p1+p2)/2 + npr.random()*(rng*2)-rng
		newpoints.append(newval)
		newpoints.append(p2)
	rng *= 2**(-ruggedness)
	return make_fractal_aux(n, newpoints, ruggedness, rng)



# return a length n random walk with each step adding Gaussian noise
#
# parameters:
#   n: the number of points on the walk
#   stdev: the size of the jumps from step to step
#   seed: the initial value along the walk
# returns:
#   a list of points mapping out the walk
#   
def make_gaussian_walk(n, stdev, seed=0):
	ys = []
	for i in range(n):
		y = seed + npr.normal(0, stdev)
		ys.append(y)
		seed = y
	return ys

