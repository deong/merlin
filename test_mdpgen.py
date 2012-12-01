#!/usr/bin/env python
#
# Unit tests for the multi-task reinforcement learning problem generator
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

import unittest
import mdpgen as gen
import math
import numpy as np
import numpy.testing as nptest
import scipy.linalg as sla
import scipy.stats as sst

class TestRandRewards1(unittest.TestCase):

	# just a basic test of the method
	def test_rewards1(self):
		numStates = 1000
		numActions = 5
		R = np.asarray([[ 1.0,	0.4, -0.4],
						[ 0.4,	1.0,  0.6],
						[-0.4,	0.6,  1.0]])
		D = gen.randinst1(numStates, numActions, R)
		self.checkCorrelations(R, D)
	
	# another basic test
	def test_rewards2(self):
		numStates = 5000
		numActions = 20
		R = np.asarray([[ 1.0, -0.7, -0.5],
						[-0.7,	1.0,  0.8],
						[-0.5,	0.8,  1.0]])
		D = gen.randinst1(numStates, numActions, R)
		self.checkCorrelations(R, D)
	
	# moving to four tasks
	def test_rewards3(self):
		numStates = 200
		numActions = 8
		R = np.asarray([[ 1.0,	0.2, -0.5,	0.0],
						[ 0.2,	1.0,  0.4,	0.0],
						[-0.5,	0.4,  1.0,	0.6],
						[ 0.0,	0.0,  0.6,	1.0]])
		D = gen.randinst1(numStates, numActions, R)
		self.checkCorrelations(R, D)
	
	# pick an invalid covariance matrix (not positive definite)
	# should throw an exception
	def test_invalidR(self):
		numStates = 1000
		numActions = 10
		R = np.asarray([[ 1.0, -0.7,  0.8],
						[-0.7,	1.0,  0.9],
						[ 0.8,	0.9,  1.0]])
		self.assertRaises(sla.LinAlgError, gen.randinst1, numStates, numActions, R)
		

	# note the number of digits of precision is taken as log10(0.2)
	# to yield a tolerance of 0.1 in the comparison method used by
	# assert_array_almost_equal
	#
	# this test will sometimes yield false positives -- failing tests
	# for correct code. I've set this tolerance fairly wide to give
	# some wiggle room in the comparison, but it isn't foolproof.
	def checkCorrelations(self, R, data):
		r = self.correlation(data)
		#nptest.assert_array_almost_equal(R, r, decimal=math.log10(0.2))
		nptest.assert_array_almost_equal(R, r, decimal=1)


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
	def correlation(self, D):
		n, m, k = D.shape
		corr = np.zeros([k, k])
		for i in range(0, k):
			for j in range(0, k):
				x = np.reshape(D[:,:,i], n*m)
				y = np.reshape(D[:,:,j], n*m)
				corr[i, j] = (sst.pearsonr(x,y)[0])
		return corr



class TestRandomGraph1(unittest.TestCase):

	def test_outdegree1(self):
		numStates = 100
		numActions = 4
		G = gen.randgraph1(numStates, numActions)
		for node in G:
			succ = [y for (x,y) in G.edges() if x==node]
			self.assertEqual(len(succ), numActions)

	def test_outdegree2(self):
		numStates = 2000
		numActions = 10
		G = gen.randgraph1(numStates, numActions)
		for node in G:
			succ = [y for (x,y) in G.edges() if x==node]
			self.assertEqual(len(succ), numActions)


if __name__ == '__main__':
	unittest.main()

