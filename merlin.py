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

from __future__ import print_function
import sys
import argparse
import ast
import numpy as np
import networkx as nx
import merlin.rewards as rwd
import merlin.graphs as grp
import merlin.gridworld as grd
import merlin.io as io
import cPickle

def demo_rand_graph_uniform_degree():
	R = np.asarray([[ 1.0,	0.4, -0.4],
					[ 0.4,	1.0,  0.6],
					[-0.4,	0.6,  1.0]])
	
	nstates = 20
	nactions = 4
	mu = [100.0] * 3
	sigma = [10.0] * 3
	cov = rwd.cor2cov(R, sigma)
	rewards = rwd.mvnrewards(nstates, nactions, mu, cov)
	G = grp.rand_graph_uniform_degree(nstates, nactions)
	cc = nx.strongly_connected_components(G)
	G2 = grp.make_strongly_connected(G)
	cc2 = nx.strongly_connected_components(G2)
	return (G, G2, cc, cc2, rewards)

def demo_maze():
	R = np.asarray([[ 1.0,	0.4, -0.4],
					[ 0.4,	1.0,  0.6],
					[-0.4,	0.6,  1.0]])
	
	mu = [100.0] * 3
	sigma = [10.0] * 3
	cov = rwd.cor2cov(R, sigma)
	z = grd.make_multimaze(10, 10, 3)
	goals = grd.maze_goal_states(z, 3, mu, cov)
	return (z, goals)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(	  '--demo',						 help='run with a sample set of parameters', action='store_true')
	parser.add_argument('-t', '--type',		  default=None,	 help='problem instance type (required)', choices=['garnet', 'randgraph', 'maze', 'nnet', 'fuzzed'])
	parser.add_argument('-n', '--states',	  default=100,	 help='size of the state space', type=int)
	parser.add_argument('-m', '--actions',	  default=4,	 help='number of available actions', type=int)
	parser.add_argument('-k', '--tasks',	  default=2,	 help='number of concurrent tasks', type=int)
	parser.add_argument('-c', '--correlation',				 help='task correlation matrix (in Python nested list form)')
	parser.add_argument('-r', '--rmeans',                    help='mean values for each task reward (in python list form)')
	parser.add_argument('-s', '--stdev',					 help='standard deviations of task rewards (in python list form)')
	parser.add_argument('-x', '--rows',		  default=10,	 help='rows in random maze', type=int)
	parser.add_argument('-y', '--cols',		  default=10,	 help='columns in random maze', type=int)
	parser.add_argument('-d', '--dimensions', default=2,     help='dimensionality of the state vector', type=int)
	parser.add_argument(      '--hidden',                    help='number of hidden units in the approximation network', type=int)
	parser.add_argument(      '--write-dot',  default=False, help='write a dot file to visualize the transition dynamics', action='store_true')
	parser.add_argument(      '--train-log',  default=False, help='write a log file of training data and predictions', action='store_true')
	parser.add_argument(      '--baseline',                  help='filename containing a trained dynamics network')
	parser.add_argument(      '--fuzz-frac',  default=0.05,  help='fraction of network weights to alter', type=float)
	parser.add_argument(      '--fuzz-scale', default=1.0,   help='amount to alter the chosen network weights by', type=float)
	args = parser.parse_args()


	if args.demo:
		# testing mazes
		# (z, goals) = demo_maze()
		# write_maze_instance(z, goals)
		# sys.exit(0)
		# end testing
		
		# testing random graphs
		# G, G2, cc, cc2, rewards = demo_rand_graph_uniform_degree()
		# print('{} components: {}'.format(len(cc), cc))
		# print('{} components: {}'.format(len(cc2), cc2))
		# sys.exit(0)
		# end testing

		# testing continuous mdp generation
		G, G2, cc, cc2, rewards = demo_rand_graph_uniform_degree()
		(nnet, svm, avm) = grp.make_continuous_mdp(G2, rewards, 2)
		grp.output_dot(G, svm, avm, 'discreteG.dot')
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
		
	# get the target mean vector for rewards. If not given, assume zero means
	if not args.rmeans:
		args.rmeans = np.zeros(args.tasks)
	else:
		args.rmeans = np.asarray(ast.literal_eval(args.rmeans))
		
	# read standard deviation for the rewards for each task. If not given, assume
	# unit standard deviations
	if not args.stdev:
		args.stdev = np.ones(args.tasks)
	else:
		args.stdev = np.asarray(ast.literal_eval(args.stdev))
		
	# compute a covariance matrix from the correlation matrix and standard deviations
	cov = rwd.cor2cov(args.correlation, args.stdev)
		
	if args.type == 'randgraph':
		rewards = rwd.mvnrewards(args.states, args.actions, args.rmeans, cov)
		transition_graph = grp.rand_graph_uniform_degree(args.states, args.actions)
		io.write_instance(transition_graph, rewards)
		print('# type={}, states={}, actions={}, correlation={}, stdev={}'.
			  format(args.type, args.states, args.actions, args.correlation.tolist(), args.stdev.tolist()))

	elif args.type == 'maze':
		maze = grd.make_multimaze(args.rows, args.cols, args.tasks)
		goals = grd.maze_goal_states(maze)
		# transition_graph = maze_transition_graph(maze, goals)
		# rewards = np.zeros([args.rows * args.cols, 4, args.tasks])
		# write_instance(transition_graph, rewards)
		io.write_maze_instance(maze, goals)
		print('# type={}, rows={}, cols={}, correlation={}, stdev={}'.
			  format(args.type, args.rows, args.col, args.correlation.tolist(), args.stdev.tolist()))

	elif args.type == 'garnet':
		# generate garnet problem
		print('todo')
		
	elif args.type == 'nnet':
		rewards = rwd.mvnrewards(args.states, args.actions, args.rmeans, cov)
		G = grp.rand_graph_uniform_degree(args.states, args.actions)
		cc = nx.strongly_connected_components(G)
		if len(cc) > 1:
			G = grp.make_strongly_connected(G)

		if args.train_log:
			args.train_log = 'train_log.dat'

		hidden_units = (args.dimensions + 2) * (args.dimensions + 2)
		if args.hidden:
			hidden_units = int(args.hidden)
				
		(nnet, traindata, svm, avm) = grp.make_continuous_mdp(G, rewards, args.dimensions, hidden_units)
		io.write_neural_net(nnet, traindata, 'dynamics.net')
		io.write_train_log(nnet, traindata, args.train_log)
			
		if args.write_dot:
			io.output_dot(G, svm, avm, 'transdyn.dot')

	elif args.type == 'fuzzed':
		if not args.baseline:
			parser.print_help()
			sys.exit(1)
		else:
			(net, trainset) = io.read_neural_net(args.baseline)
			net2 = grp.fuzz_neural_net(net, args.fuzz_frac, args.fuzz_scale)
			io.write_neural_net(net2, trainset, 'fuzzed.net')
			io.write_train_log(net2, trainset, 'train_log_fuzzed.dat')
			
	else:
		print('invalid problem type specified: {}', args.type)
		parser.print_help()
		sys.exit(1)
		

