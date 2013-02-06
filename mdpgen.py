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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import random
import numpy as np
import scipy.linalg as sla
import numpy.random as npr
import networkx as nx



#
# create random reward structure for an (nstates, nactions) MDP
#
# parameters:
#   nstates:     number of states
#   nactions: number of actions
#   covmat:  nxn covariance matrix, where n=number of tasks
# returns:
#   an (nstates x nactions x ntasks) reward matrix
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
    while total_in > sum(dout):
        node = random.randint(0, nodes - 1)
        din[node] += 1
        total_in += 1

    # if we did this right, the sums should be guaranteed to match
    assert(sum(din) == sum(dout))

    # generate a random directed multigraph with the specified degree
    # sequences
    tgraph = nx.directed_configuration_model(din, dout)
    return tgraph




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
#   G: state transition graph
#   R: reward structure
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
                    line += "{0:.3f} ".format(D[node, index, task])
        print(line)


if __name__ == '__main__':
    states = 100
    actions = 4
    tasks = 3
    R = np.asarray([[ 1.0,  0.4, -0.4],
                    [ 0.4,  1.0,  0.6],
                    [-0.4,  0.6,  1.0]])
    D = randinst1(states, actions, R)
    G = randgraph1(states, actions)
    write_instance(G, D)
