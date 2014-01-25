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


import sys
import argparse
import ast
import random
import numpy as np
import scipy.linalg as sla
import numpy.random as npr
import networkx as nx
import math
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
# create a random graph with uniform out-degree
#
# parameters:
#	nstates: number of states
#	nactions: number of actions
# returns:
#	a directed multigraph with each node having out-degree exactly
#	equal to edges and in-degree > 0.
#
def rgud(nodes, edges):
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
	return tgraph


#
# take an arbitrary directed graph and make it strongly connected,
# maintaining the total number of incoming and outgoing edges for
# each vertex.
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
	max_attempts = num_components * math.log2(num_components)
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

			

# construct a new MDP given a set of rewards and a transition graph
# and write it to stdout
#
# the format of the output records is as follows
#
# -------------------------------------------------------------
# numStates numActions numTasks
#
# 0 [successor [reward_i]{numTasks}]{numActions}
# ...
# numStates-1 [successor [reward_i]{numTasks}]{numActions}
# -------------------------------------------------------------
#
# parameters:
#	G: state transition graph
#	R: reward structure
#
def write_instance(G, R):
	n, m, k = R.shape
	# number of nodes in the transition graph should be equal to
	# the number of states in the reward matrix
	assert(G.number_of_nodes() == n)
	print("{} {} {}\n".format(n, m, k))
	for node in G:
		line = "{} ".format(node)
		for index, edge in enumerate(G.successors(node)):
			# note that the enumeration flattens out any duplicated
			# edges; dups are fine for MDPs -- they just indicate two
			# actions that lead to the same successor state. So we
			# compensate for this by calculating the number of dups
			# and explicitly repeating them the right number of times
			for i in range(0, len(G[node][edge])):
				line += "{} ".format(edge)
				for task in range(0, k):
					line += "{0:.3f} ".format(R[node, index, task])
		print(line)
	print("\n")



# create a smooth "landscape" of real values on a mesh
#
# For the neural network-based continuous MDP generation, we need to associate
# a real-value with each node in the network. Doing this completely randomly
# results in a very rugged transition graph -- moving from one state to an
# adjacent state often results in a very large jump in the real-valued state
# space, which is both physically unrealistic for most applications, and also
# very difficult for the network to learn to approximate.
#
# Instead, we want to generate a procedural "landscape" so that adjacent values
# are roughly correlated.
#
# parameters:
#   G: state transition graph
#   
def generate_landscape(G):
	return

# return a 1-D fractal landscape generated using the midpoint displacement method
def make_fractal(n, ruggedness, rng=1.0, seed=0):
	points = [npr.random(), npr.random()]
	return make_fractal_aux(n, points, ruggedness, rng)

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
def make_gaussian_walk(n, stdev, seed=0):
	ys = []
	for i in range(n):
		y = seed + npr.normal(0, stdev)
		ys.append(y)
		seed = y
	return ys



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
#   
def make_continuous_mdp(G, R, inpd):
	# build up a training set for the neural network; input is each component
	# of the current state vector + one action, and output is the components
	# of the successive state vector
	training_set = pybrain.datasets.SupervisedDataSet(inpd + 1, inpd)

	# first, assign a random vector to each state/action pair in the graph
	# using the fractal landscape method
	num_points = len(G)
	state_values = []
	for d in range(inpd):
		vals = make_fractal(num_points, 0.75)
		state_values.append(vals)
		print(state_values)
		print('\n\n')
	state_values = list(zip(*state_values))
	print(state_values)
	print('{}x{}'.format(len(state_values), len(state_values[0])))
		  
	state_value_map = {}
	for i, node in enumerate(G):
		state_value_map[node] = state_values[i]

	# now go back through the graph, for each node connecting it via an action
	# to a successor node
	print("training_set")
	for node in G:
		for index, succ in enumerate(G.successors(node)):
			inp  = np.append(state_values[node], npr.random() * 2.0 - 1.0)
			outp = state_values[succ]
			training_set.addSample(inp, outp)
			print(inp)
			print(outp)

	# finally, create a train a network
	# hidden units:
	#    1 * inpd = very fast, not much improvement
	#    2 * inpd = 5+ minutes, some improvement
	#    3 * inpd = 6 minutes, some improvement
	#    4 * inpd = 7.5 seconds
	nnet = pybrain.tools.shortcuts.buildNetwork(inpd + 1, 4 * inpd, inpd, bias=True)
	trainer = pybrain.supervised.trainers.BackpropTrainer(nnet, training_set)
	errors = trainer.trainEpochs(1000)
	#errors = trainer.trainUntilConvergence()
	# print("training_errors: " + errors)

	print("output: ")
	for point in training_set:
		outp = nnet.activate(point[0])
		print(point)
		print(outp)

	return nnet
		



										   
#
# Maze-type problems (gridworld)
#
# The basic structure of this type of problem is that each is a 2-d world consisting
# of separate "trails" for each task. Note that there are generally no walls between
# these trails, so an agent is free to move through the world as it chooses. The trails
# simply imply that each cell in the world is marked with a task number, and if the
# agent is in a cell with task number X, there is a path leading to the goal state for
# task X through which all intermediate cells are also marked X.
#
# The transition dynamics are simple up, down, left, right actions for each state. If
# you are at a boundary cell, attempting to move out of the world results in a negative
# penalty for each task and leaves the agent in the same state. This is true even if
# the current state is a goal state.
#


#
# Multi-Maze Generator using Depth-first Search
# Multi-Maze: Maze w/ multiple paths to solve
#
# Code adapted from 
# http://code.activestate.com/recipes/578378-random-multi-maze-generator/
# Available under MIT License.
#
# The output of this function is a 2d matrix structure, where the (i,j) pair
# is the "path number", i.e., maze[i][j] = k if cell (i,j) in the maze is
# used by the k-th distinct path through the maze.
# 
def make_multimaze(width, height, nTasks):
	# width and height of the maze
	mx = width	
	my = height 

	# 4 directions to move in the maze
	dx = [0, 1, 0, -1] 
	dy = [-1, 0, 1, 0] 

	maze = [[0 for x in range(mx)] for y in range(my)]

	stack = [] # array of stacks
	for i in range(nTasks):
		while True:
			kx = random.randint(0, mx - 1); ky = random.randint(0, my - 1)
			if maze[ky][kx] == 0: break
		stack.append([(kx, ky)])
		maze[ky][kx] = i + 1

	cont = True # continue
	while cont:
		cont = False
		for p in range(nTasks):
			if len(stack[p]) > 0:
				cont = True # continue as long as there is a non-empty stack
				(cx, cy) = stack[p][-1]
				# find a new cell to add
				nlst = [] # list of available neighbors
				for i in range(4):
					nx = cx + dx[i]
					ny = cy + dy[i]
					if nx >= 0 and nx < mx and ny >= 0 and ny < my:
						if maze[ny][nx] == 0:
							# of occupied neighbors must be 1
							ctr = 0
							for j in range(4):
								ex = nx + dx[j]; ey = ny + dy[j]
								if ex >= 0 and ex < mx and ey >= 0 and ey < my:
									if maze[ey][ex] == p + 1: ctr += 1
							if ctr == 1: nlst.append(i)
				# if 1 or more neighbors available then randomly select one and add
				if len(nlst) > 0:
					ir = nlst[random.randint(0, len(nlst) - 1)]
					cx += dx[ir]; cy += dy[ir]
					maze[cy][cx] = p + 1
					stack[p].append((cx, cy))
				else: stack[p].pop()
	return np.asarray(maze)


#
# create a reward structure for a maze
#
# The basic idea is that there should be a separate location in the
# maze with a positive reward for each task. As the maze generator
# produces separate paths, we should aim to put the reward for each
# task somewhere along that task's path.
#
# returns an array of the same size and shape as the maze, but with
# zeros everywhere except for $tasks non-zero entries.
#
def maze_goal_states(maze, tasks, mu, cov):
	rows = maze.shape[0]
	cols = maze.shape[1]
	reward_values = npr.multivariate_normal(mu, cov)
	goals = np.zeros([tasks, rows, cols])
	
	# for each task, build a list of maze locations with that task id;
	# choose one at random to be the selected goal state for that task
	for task in range(tasks):
		locs = np.transpose(np.where(maze == (task+1)))
		goal_loc = locs[npr.randint(0, len(locs))]
		goals[task, goal_loc[0], goal_loc[1]] = reward_values[task]
	return goals


#
# write an instance of the multimaze problem
#
# Ideally, this should be unified with the graph instances, but the
# way that duplicate edges are handled in the graph-based instances
# loses information that is important for mazes (it mixes up which
# action is which). For now I handle this by using a custom writer
# for the mazes.
#
def write_maze_instance(maze, goals):
	tasks, rows, cols = goals.shape
	print("{} {} {}\n".format(rows*cols, 4, tasks))

	for row in range(rows):
		for col in range(cols):
			node_num = rowcol_to_index(maze, row, col)
			line = "{} ".format(node_num)

			# order of actions is up, down, left, right
			neighbors = [(x, col) for x in [row-1, row+1]] + [(row, y) for y in [col-1, col+1]]
			for action, (x,y) in enumerate(neighbors):
				target = rowcol_to_index(maze, x, y)
				if target != None:
					line += "{} ".format(target)
					for task in range(tasks):
						line += "{} ".format(goals[task, x, y])
				else:
					line += "{} ".format(node_num)
					for task in range(tasks):
						line += "{} ".format(-10)
			print(line)
	print("\n")


#
# take a maze, row, and column, and return a node number or None
# if the requested row and column are out of bounds
#
def rowcol_to_index(maze, row, col):
	rows, cols = maze.shape
	if row < 0 or row >= rows or col < 0 or col >= cols:
		return None
	index = row*maze.shape[0] + col
	if index >= maze.size or index < 0:
		return None
	else:
		return index



#
# convert a correlation matrix to a covariance matrix with given standard deviations
#
def cor2cov(R, sigma):
	return np.diag(sigma).dot(R).dot(np.diag(sigma))



def demo_rgud():
	R = np.asarray([[ 1.0,	0.4, -0.4],
					[ 0.4,	1.0,  0.6],
					[-0.4,	0.6,  1.0]])
	
	nstates = 50
	nactions = 2
	mu = [100.0] * 3
	sigma = [10.0] * 3
	cov = cor2cov(R, sigma)
	rewards = mvnrewards(nstates, nactions, mu, cov)
	G = rgud(nstates, nactions)
	cc = nx.strongly_connected_components(G)
	G2 = make_strongly_connected(G)
	cc2 = nx.strongly_connected_components(G2)
	return (G, G2, cc, cc2, rewards)

def demo_maze():
	R = np.asarray([[ 1.0,	0.4, -0.4],
					[ 0.4,	1.0,  0.6],
					[-0.4,	0.6,  1.0]])
	
	mu = [100.0] * 3
	sigma = [10.0] * 3
	cov = cor2cov(R, sigma)
	z = make_multimaze(10, 10, 3)
	goals = maze_goal_states(z, 3, mu, cov)
	return (z, goals)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(	  "--demo",						 help="run with a sample set of parameters", action='store_true')
	parser.add_argument("-t", "--type",		  default=None,	 help="problem instance type {rgudcr,rzcgl}")
	parser.add_argument("-n", "--states",	  default=100,	 help="size of the state space", type=int)
	parser.add_argument("-m", "--actions",	  default=4,	 help="number of available actions", type=int)
	parser.add_argument("-k", "--tasks",	  default=2,	 help="number of concurrent tasks", type=int)
	parser.add_argument("-c", "--correlation",				 help="task correlation matrix (in Python nested list form)")
	parser.add_argument("-s", "--stdev",					 help="standard deviations of task rewards (in python list form)")
	parser.add_argument("-x", "--rows",		  default=10,	 help="rows in random maze", type=int)
	parser.add_argument("-y", "--cols",		  default=10,	 help="columns in random maze", type=int)
	
	
	args = parser.parse_args()


	if args.demo:
		# testing mazes
		# (z, goals) = demo_maze()
		# write_maze_instance(z, goals)
		# sys.exit(0)
		# end testing
		
		# testing random graphs
		# G, G2, cc, cc2, rewards = demo_rgud()
		# print("{} components: {}".format(len(cc), cc))
		# print("{} components: {}".format(len(cc2), cc2))
		# sys.exit(0)
		# end testing

		# testing continuous mdp generation
		G, G2, cc, cc2, rewards = demo_rgud()
		nnet = make_continuous_mdp(G2, rewards, 2)
		print(nnet)
		sys.exit(0)
		# end testing
		
	
	if not args.type:
		parser.print_help()
		sys.exit(1)

	# read correlation matrix from command line argument. If not given, assume
	# independent tasks
	if not args.correlation:
		args.correlation = np.identity(args.tasks)
	else:
		args.correlation = np.asarray(ast.literal_eval(args.correlation))
		
	# read standard deviation for the rewards for each task. If not given, assume
	# unit standard deviations
	if not args.stdev:
		args.stdev = np.ones(args.tasks)
	else:
		args.stdev = np.asarray(ast.literal_eval(args.stdev))
		
	# compute a covariance matrix from the correlation matrix and standard deviations
	cov = cor2cov(args.correlation, args.stdev)
		
	if args.type == "rgudcr":
		mu = [0.0] * args.tasks
		rewards = mvnrewards(args.states, args.actions, mu, cov)
		transition_graph = rgud(args.states, args.actions)
		write_instance(transition_graph, rewards)
		print("# type={}, states={}, actions={}, correlation={}, stdev={}".
			  format(args.type, args.states, args.actions, args.correlation.tolist(), args.stdev.tolist()))
	elif args.type == "rzcgl":
		maze = make_multimaze(args.rows, args.cols, args.tasks)
		goals = maze_goal_states(maze)
		# transition_graph = maze_transition_graph(maze, goals)
		# rewards = np.zeros([args.rows * args.cols, 4, args.tasks])
		# write_instance(transition_graph, rewards)
		write_maze_instance(maze, goals)
		print("# type={}, rows={}, cols={}, correlation={}, stdev={}".
			  format(args.type, args.rows, args.col, args.correlation.tolist(), args.stdev.tolist()))
	else:
		print("invalid problem type specified: {}", args.type)
		parser.print_help()
		sys.exit(1)
		
	# R = np.asarray([[ 1.0,  0.4, -0.4],
	#				  [ 0.4,  1.0,	0.6],
	#				  [-0.4,  0.6,	1.0]])
	# R = np.asarray([[ 1.0,  0.4, -0.4,  0.0],
	#				  [ 0.4,  1.0,	0.6, -0.1],
	#				  [-0.4,  0.6,	1.0,  0.2],
	#				  [ 0.0, -0.1,	0.2,  1.0]])
	# R = np.asarray([[1.0, 0.0],
	#				  [0.0, 1.0]])


