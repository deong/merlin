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
import numpy.random as npr


# map a real-valued state vector onto each node in the given transition graph
#
# parameters:
#   G: the state transition graph
#   inpd: the desired dimensionality of the state space
#   func_type: the type of landscape to generate {"fractal", "gaussian"}
#   ruggedness: a parameter governing the ruggedness of the graph
def make_state_value_map(G, inpd, func_type, ruggedness):
	if func_type == 'fractal':
		func = make_fractal
	elif func_type == 'gaussian':
		func = make_gaussian_walk
	else:
		raise RuntimeError('illegal state-value function type "{}" given'.format(func_type))
	
	# first, assign a random vector to each state/action pair in the graph
	# using the fractal landscape method
	num_points = len(G)
	state_values = []
	for d in range(inpd):
		vals = func(num_points, ruggedness)
		state_values.append(vals)
	state_values = list(zip(*state_values))
	
	# create a map from node to state vector	  
	state_value_map = {}
	for i, node in enumerate(G):
	 	state_value_map[node] = state_values[i]

	return state_value_map


# map a real-valued action value onto each edge in the given transition graph
#
# tries to evenly distribute the action values in the allowed range so that you
# don't end up with two states combining with very similar action values mapping
# onto two very different resulting states in the transition dynamics
#
# parameters:
#   G: the state transition graph
#   action_range: a tuple of (min, max) for action values
# returns:
#   the map of actions to action values
#   
def make_action_value_map(G, action_range):
	# create a list of actions for each node
	action_value_map = {}
	for node in G:
		num_actions = len(G.successors(node))
		step = (action_range[1] - action_range[0])/2.0
		min_val = action_range[0]
		max_val = min_val + step
		for index, succ in enumerate(G.successors(node)):
			aval = npr.random() * (max_val - min_val) + min_val
			action_value_map[(node, succ)] = aval
			min_val += step
			max_val += step
	return action_value_map



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

