#!/usr/bin/env python
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# Reads in a defined environment and does a random walk through the world
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

from __future__ import print_function
import sys
import argparse
import numpy as np
import numpy.random as npr
import merlin.io as io
import merlin.net as net

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--type',		  default=None,	 help='problem instance type (required)', choices=['garnet', 'randgraph', 'maze', 'nnet', 'fuzzed', 'svm'])

	parser.add_argument('--transitions-net',                 help='filename containing a trained dynamics network')
	parser.add_argument('--rewards-net',                     help='filename containing a trained rewards network')
	parser.add_argument('--walk-len',         default=1000,  help='number of steps to take in the world', type=int)
	args = parser.parse_args()


	if not args.type:
		parser.print_help()
		sys.exit(1)

		
	if args.type == 'nnet':
		if not args.transitions_net or not args.rewards_net:
			parser.print_help()
			sys.exit(1)

		(tnet, tset) = io.read_neural_net(args.transitions_net)
		(rnet, rset) = io.read_neural_net(args.rewards_net)
		state_dim = tnet['in'].dim - 1
		reward_dim = rnet['out'].dim

		state = np.zeros(state_dim)
		for i in range(args.walk_len/2):
			action = npr.random() * 0.5 + 0.5
			new_state = net.activation(tnet, state, action)
			reward = net.activation(rnet, state, action)
			entry = state.tolist() + [action] + new_state.tolist() + reward.tolist()
			print('{}'.format(' '.join([str(x) for x in entry])))
			state = new_state
		for i in range(args.walk_len/2):
			action = npr.random() * 0.5 - 0.5
			new_state = net.activation(tnet, state, action)
			reward = net.activation(rnet, state, action)
			entry = state.tolist() + [action] + new_state.tolist() + reward.tolist()
			print('{}'.format(' '.join([str(x) for x in entry])))
			state = new_state

	elif args.type == 'svm':
		if not args.transitions_net or not args.rewards_net:
			parser.print_help()
			sys.exit(1)

		(statefuncs, statetd) = io.read_svm_model(args.transitions_net)
		(rewardfuncs, rewardtd) = io.read_svm_model(args.rewards_net)
		state_dim = len(statetd[0][0][0])-1
		reward_dim = len(rewardtd)
		state = np.zeros(state_dim)
		for i in range(args.walk_len/2):
			action = npr.random() * 0.5 + 0.5
			s_prime = []
			for dim in range(state_dim):
				s_prime.append(statefuncs[dim].predict(np.append(state, action))[0])
			s_prime = np.asarray(s_prime)

			reward = []
			for task in range(reward_dim):
				reward.append(rewardfuncs[task].predict(np.append(state, action))[0])
			reward = np.asarray(reward)
			entry = state.tolist() + [action] + s_prime.tolist() + reward.tolist()
			print('{}'.format(' '.join([str(x) for x in entry])))
			state = s_prime
		for i in range(args.walk_len/2):
			action = npr.random() * 0.5 - 0.5
			s_prime = []
			for dim in range(state_dim):
				s_prime.append(statefuncs[dim].predict(np.append(state, action))[0])
			s_prime = np.asarray(s_prime)

			reward = []
			for task in range(reward_dim):
				reward.append(rewardfuncs[task].predict(np.append(state, action))[0])
			reward = np.asarray(reward)
			entry = state.tolist() + [action] + s_prime.tolist() + reward.tolist()
			print('{}'.format(' '.join([str(x) for x in entry])))
			state = s_prime
			
