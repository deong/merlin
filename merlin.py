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
import merlin.values as values
import merlin.net as net
import cPickle



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(	  '--demo',						 help='run with a sample set of parameters', action='store_true')
	parser.add_argument('-t', '--type',		  default=None,	 help='problem instance type (required)', choices=['garnet', 'randgraph', 'maze', 'nnet', 'fuzzed', 'svm'])
	parser.add_argument('-n', '--states',	  default=100,	 help='size of the state space', type=int)
	parser.add_argument('-m', '--actions',	  default=4,	 help='number of available actions', type=int)
	parser.add_argument('-k', '--tasks',	  default=2,	 help='number of concurrent tasks', type=int)
	parser.add_argument('-c', '--correlation',				 help='task correlation matrix (in Python nested list form)')
	parser.add_argument('-r', '--rmeans',                    help='mean values for each task reward (in python list form)')
	parser.add_argument('-s', '--stdev',					 help='standard deviations of task rewards (in python list form)')
	parser.add_argument('-x', '--rows',		  default=10,	 help='rows in random maze', type=int)
	parser.add_argument('-y', '--cols',		  default=10,	 help='columns in random maze', type=int)

	# neural net parameters
	parser.add_argument('-d', '--dimensions', default=2,     help='dimensionality of the state vector', type=int)
	parser.add_argument(      '--hidden',                    help='number of hidden units in the approximation network', type=int)
	parser.add_argument(      '--rhidden',                   help='number of hidden units in the reward approximation network', type=int)
	parser.add_argument(      '--transitions-net',           help='file to save the transition dynamics network to')
	parser.add_argument(      '--rewards-net',               help='file to save the rewards network to')
	parser.add_argument(      '--transitions-dot',           help='name of the file to write the transition dynamics to in dot format')
	parser.add_argument(      '--transitions-log',           help='name of the file to write training data and predictions to')
	parser.add_argument(      '--rewards-log',               help='name of the file to write reward network training log to')
	parser.add_argument(      '--max-epochs', default=2000,  help='maximum number of training epochs for the networks', type=int)

	# fuzzed neural net arguments
	parser.add_argument(      '--baseline',                  help='filename containing a trained dynamics network')
	parser.add_argument(      '--fuzz-frac',  default=0.05,  help='fraction of network weights to alter', type=float)
	parser.add_argument(      '--fuzz-scale', default=1.0,   help='amount to alter the chosen network weights by', type=float)

	# svm-regression parameters
	parser.add_argument(      '--svm-C',      default=1.0,   help='penalty parameter for the SVM error term', type=float)
	parser.add_argument(      '--svm-epsilon',default=0.1,   help='tolerance within which no error is applied', type=float)

	args = parser.parse_args()


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
	if not rwd.is_pos_def(cov):
		print('Error: covariance matrix must be positive definite', file=sys.stderr)
		sys.exit(1)
		
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
		if not args.transitions_net:
			args.transitions_net = 'dynamics.net'
		if not args.rewards_net:
			args.rewards_net = 'rewards.net'
			
		# generate the underlying graph for the transition dynamics
		G = grp.rand_graph_uniform_degree(args.states, args.actions)
		cc = nx.strongly_connected_components(G)
		if len(cc) > 1:
			G = grp.make_strongly_connected(G)

		scalef = 10.0
		values.annotate_states_walk(G, args.dimensions, 'fractal', 0.7, scalef)
		values.annotate_actions(G, (-2.0, 2.0))
		
		hidden_units = (args.dimensions + 2) * (args.dimensions + 2)
		if args.hidden:
			hidden_units = int(args.hidden)

		print('Training neural network on state dynamics...this may take a while...', file=sys.stderr)
		(nnet, traindata) = grp.make_continuous_mdp(G, args.dimensions, hidden_units, args.max_epochs)
		io.write_neural_net(nnet, traindata, args.transitions_net)
		if args.transitions_log:
			io.write_train_log(nnet, traindata, args.transitions_log)

		# generate the correlated rewards and a network predicting them
		rewards = rwd.mvnrewards(args.states, args.actions, args.rmeans, cov)
		rwd.annotate_rewards(G, rewards)
		
		if not args.rhidden:
			args.rhidden = (args.dimensions + 2) * (args.tasks + 2)

		print('Training neural network on reward function...this may take a while...', file=sys.stderr)
		(reward_net, reward_data) = rwd.learn_reward_function(G, args.rhidden, args.max_epochs)
		io.write_neural_net(reward_net, reward_data, args.rewards_net)
		if args.rewards_log:
			io.write_train_log(reward_net, reward_data, args.rewards_log)
		
		if args.transitions_dot:
			io.output_dot(G, args.transitions_dot)

	elif args.type == 'fuzzed':
		if not args.baseline:
			parser.print_help()
			sys.exit(1)
		else:
			(net, trainset) = io.read_neural_net(args.baseline)
			net2 = net.fuzz_neural_net(net, args.fuzz_frac, args.fuzz_scale)
			io.write_neural_net(net2, trainset, 'fuzzed.net')
			io.write_train_log(net2, trainset, 'train_log_fuzzed.dat')

	elif args.type == 'svm':
		if not args.transitions_net:
			args.transitions_net = 'dynamics.svm'
		if not args.rewards_net:
			args.rewards_net = 'rewards.svm'
			
		# generate the underlying graph for the transition dynamics
		G = grp.rand_graph_uniform_degree(args.states, args.actions)
		cc = nx.strongly_connected_components(G)
		if len(cc) > 1:
			G = grp.make_strongly_connected(G)

		scalef = 10.0
		values.annotate_states_walk(G, args.dimensions, 'fractal', 0.7, scalef)
		values.annotate_actions(G, (-2.0, 2.0))

		# generate the correlated rewards and a network predicting them
		rewards = rwd.mvnrewards(args.states, args.actions, args.rmeans, cov)
		rwd.annotate_rewards(G, rewards)
	
		models = []
		training_sets = []
		for dim in range(args.dimensions):
			print('building regression model for S_{}...'.format(dim))
			model, training_data = grp.build_regression_model(G, dim, args.svm_C, args.svm_epsilon)
			models.append(model)
			training_sets.append(training_data)
		io.write_svm_model(models, training_sets, args.transitions_net)

		rmodels = []
		rtraining_sets = []
		for task in range(args.tasks):
			print('building regression model for R_{}...'.format(task))
			model, training_data = rwd.learn_reward_function_svm(G, task, args.svm_C, args.svm_epsilon)
			rmodels.append(model)
			rtraining_sets.append(training_data)
		io.write_svm_model(rmodels, rtraining_sets, args.rewards_net)
			
		if args.transitions_log:
			io.write_svm_train_log(models, training_sets, args.transitions_log)
		if args.rewards_log:
			io.write_svm_train_log(rmodels, rtraining_sets, args.rewards_log)
		
	else:
		print('invalid problem type specified: {}', args.type)
		parser.print_help()
		sys.exit(1)
		


