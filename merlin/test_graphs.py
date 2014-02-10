#!/usr/bin/env python
#
# merlin/test_graphs.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merlin, the generator for multitask environments
# for reinforcement learners.
#
# Unit tests for the multi-task reinforcement learning problem generator
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

import unittest
import merlin.graphs as graphs
import merlin.rewards as rewards
import merlin.values as values
import merlin.gridworld as grid
import math
import numpy as np
import numpy.random as npr
import numpy.testing as nptest
import scipy.linalg as sla
import scipy.stats as sst
import networkx as nx



# Test cases for generating random transition graphs with uniform out-degree
class TestRGUD(unittest.TestCase):

    # test whether some random graphs have the correct number of outgoing
    # edges for each node
    def test_outdegree1(self):
        numStates = 100
        numActions = 4
        G = graphs.rand_graph_uniform_degree(numStates, numActions)
        for node in G:
            succ = [y for (x,y) in G.edges() if x==node]
            self.assertEqual(len(succ), numActions)
    
    def test_outdegree2(self):
        numStates = 2000
        numActions = 10
        G = graphs.rand_graph_uniform_degree(numStates, numActions)
        for node in G:
            succ = [y for (x,y) in G.edges() if x==node]
            self.assertEqual(len(succ), numActions)
    
    def test_outdegree3(self):
        numStates = 200
        numActions = 2
        G = graphs.rand_graph_uniform_degree(numStates, numActions)
        for node in G:
            succ = [y for (x,y) in G.edges() if x==node]
            self.assertEqual(len(succ), numActions)

    # this one has way more edges than nodes, so there must be loads of
    # redundant edges
    def test_outdegree4(self):
        numStates = 50
        numActions = 1000
        G = graphs.rand_graph_uniform_degree(numStates, numActions)
        for node in G:
            succ = [y for (x,y) in G.edges() if x==node]
            self.assertEqual(len(succ), numActions)

    # test whether some random graphs are strongly connected
    def test_connectedness(self):
        ntests = 50
        nsuccess = 0
        for test in range(ntests):
            numStates = npr.randint(100, 5000)
            numActions = npr.randint(2, 20)
            G = graphs.make_strongly_connected(graphs.rand_graph_uniform_degree(numStates, numActions))
            if nx.number_strongly_connected_components(G) == 1:
                nsuccess += 1
        self.assertEqual(nsuccess, ntests)
            


if __name__ == '__main__':
    unittest.main()

