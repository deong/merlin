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
import sklearn.svm as svm
import sklearn.gaussian_process as gp

#
# create a random graph with uniform out-degree
#
# parameters:
#   nstates: number of states
#   nactions: number of actions
# returns:
#   a directed multigraph with each node having out-degree exactly
#   equal to edges and in-degree > 0.
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

    # make sure all nodes have the correct number of outgoing edges
    for node in tgraph:
        assert(edges == len(tgraph.out_edges(node)))

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
    num_edges = len(G.out_edges(G.nodes()[0]))
    
    components = list(nx.strongly_connected_components(G))
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
        components = list(nx.strongly_connected_components(G))
        num_components = len(components)
        attempts += 1

    # make sure all nodes have the correct number of outgoing edges
    for node in G:
        assert(num_edges == len(G.out_edges(node)))
        
    return G



# make a complete lobster graph with exactly n nodes along the spine
#
# parameters:
#   nodes: number of nodes along the spine
#   edges: (ignored)
#
# adapted from the networkx random_lobster method to remove the stochasticity
# in the length of the spine
def complete_lobster(n, edges, seed=None):
    if seed is not None:
        random.seed(seed)
    p1 = p2 = 1.0
    llen=n
    L=nx.path_graph(llen)
    L.name="random_lobster(%d,%s,%s)"%(n,p1,p2)
    # build caterpillar: add edges to path graph with probability p1
    current_node=llen-1
    for n in xrange(llen):
        if npr.random()<p1: # add fuzzy caterpillar parts
            current_node+=1
            L.add_edge(n,current_node)
            if random.random()<p2: # add crunchy lobster bits
                current_node+=1
                L.add_edge(current_node-1,current_node)

    g = nx.MultiDiGraph(L) # voila, un lobster!
    for node in g:
        # if it's a leaf node, add a self edge and an edge back to the spine
        if len(g.out_edges(node)) == 1:
            knee = g.in_edges(node)[0][0]
            knee_edges = sorted(g.in_edges(knee))
            spine = knee_edges[0][0]
            g.add_edge(node, spine)
        if len(g.out_edges(node)) == 2:
            g.add_edge(node, node)
    return g


# make a random lobster graph converted to a directed multi-graph
#
# parameters:
#   nodes: expected number of nodes along the spine of the lobster
#   edges: number of edges (ignored and forced to 3)
#
# Note: for all non-spine nodes, we add edges to make the out-degree uniform.
# One edge is added as a self-edge. If another is needed (for leaf nodes, it is
# added to its predecessor node.
def random_lobster(nodes, edges):
    g = nx.MultiDiGraph(nx.random_lobster(nodes, 1.0, 1.0))
    for node in g:
        # if it's a leaf node, add a self edge and an edge back to the spine
        if len(g.out_edges(node)) == 1:
            knee = g.in_edges(node)[0][0]
            spine = g.in_edges(knee)[0][0]
            g.add_edge(node, spine)
        if len(g.out_edges(node)) == 2:
            g.add_edge(node, node)
    return g


# return a random fern graph
#
# parameters:
#   nodes: number of nodes along the spine
#   edges: (ignored)
#
# adapted from the networkx random_lobster method to remove the stochasticity
# in the length of the spine
def random_fern(backbone_len, p, frond_size, frond_actions, seed=None):
    if seed is not None:
        random.seed(seed)
    L=nx.path_graph(backbone_len)
    g = nx.MultiDiGraph(L) # voila, un lobster!

    # build fern: along the backbone, add links (with probability p)
    # to entire random subgraphs that are otherwise unconnected to the
    # spine or each other.
    for i in xrange(backbone_len-1):
        if random.random()>=p:
            continue
        next_node_index = g.number_of_nodes()
        frond = rand_graph_uniform_degree(frond_size, frond_actions)
        frond = make_strongly_connected(frond)
        g = nx.disjoint_union(g, frond)
        g.add_edge(i, next_node_index)
        g.add_edge(next_node_index, i)

        # somehow it's possible to get nodes with no incoming edges.
        # fix that by checking connectivity to each newly added node
        # from the 0 node, and add edges if necessary.
        for node in xrange(next_node_index, g.number_of_nodes()):
            if not nx.has_path(g, 0, node):
                g.add_edge(node-1, node)
                # try to delete a redundant edge from the parent
                if g.has_edge(node-1, node-1):
                    g.remove_edge(node-1, node-1)
                    
    for node in g:
        # if it's a leaf node, add a self edge and an edge back to the spine
        if len(g.out_edges(node)) == 1:
            knee = g.in_edges(node)[0][0]
            knee_edges = sorted(g.in_edges(knee))
            spine = knee_edges[0][0]
            if not g.has_edge(node,spine):
                g.add_edge(node, spine)
        if len(g.out_edges(node)) == 2:
            if not g.has_edge(node, node):
                g.add_edge(node, node)
    return g


# return a random graph of the requested type
#
# parameters:
#   gtype: the type of graph to create
#   n:     number of nodes in the graph
#   m:     number of edges from each node
#   params: dictionary of additional parameters
#
def create_graph(gtype, n, m, params):
    if gtype == 'random':
        g = rand_graph_uniform_degree(n, m)
        g = make_strongly_connected(g)
        return g
    elif gtype == 'lobster':
        # TODO: allow better control over the graph structure in lobster graphs
        g = complete_lobster(n, m)
        return g
    elif gtype == 'fern':
        g = random_fern(n, params['frond_probability'],
                        params['frond_size'], params['frond_actions'])
        return g


    
# construct a continuous MDP based on a random graph
#
# The basic idea is to construct a random graph with the desired properties,
# then fit each node and edge with a real number and train a neural network to
# predict the new state value given a state/action pair.
#
# parameters:
#   G: state transition graph
#   indim: number of state variables per state
#   hidden_units: number of hidden units in the network
#   max_epochs: maximum number of training epochs for the network
#   
def make_continuous_mdp(G, inpd, hidden_units, max_epochs):
    # build up a training set for the neural network; input is each component
    # of the current state vector + one action, and output is the components
    # of the successive state vector
    training_set = pybrain.datasets.SupervisedDataSet(inpd + 1, inpd)

    # now go back through the graph, for each node connecting it via an action
    # to a successor node
    for node in G:
        s = G.node[node]['state']
        for index, (_, succ, key) in enumerate(G.out_edges(node, keys=True)):
            a = G.edge[node][succ][key]['action']
            sp = G.node[succ]['state']
            training_set.addSample(np.append(s, a), sp)

    # finally, create a train a network
    nnet = pybrain.tools.shortcuts.buildNetwork(inpd + 1, hidden_units, inpd, bias=True)
    trainer = pybrain.supervised.trainers.BackpropTrainer(nnet, training_set)
    errors = trainer.trainUntilConvergence(maxEpochs=max_epochs)

    return (nnet, training_set)



# build an SVM regression model with the given training data
#
# parameters:
#   G: the annotated transition graph
#   state_var: the dimension in the state space to be learned
#   C: the SVM regularization parameter
#   epsilon: the tolerance outside of which the error is applied
# returns:
#   a tuple of a trained svm model and the training data
#   
def build_svm_regression_model(G, state_var, C=1.0, epsilon=0.1):
    model = svm.SVR()
    xs = []
    ys = []
    for node in G:
        for (_, succ, key) in G.out_edges(node, keys=True):
            xs.append(list(G.node[node]['state']) + [G.edge[node][succ][key]['action']])
            ys.append(G.node[succ]['state'][state_var])
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    return (model.fit(xs, ys), (xs, ys))



# build gaussian process regression model with the given training data
# 
# parameters:
#   G: the annotated transition graph
#   state_var: the dimension in the state space to be learned
#   theta0: the initial estimate for the GP parameters
# returns:
#   a tuple of a trained GP model and the training data
#   
def build_gp_regression_model(G, state_var, theta0):
    model = gp.GaussianProcess(theta0=theta0)
    xs = []
    ys = []
    for node in G:
        for (_, succ, key) in G.out_edges(node, keys=True):
            xs.append(list(G.node[node]['state']) + [G.edge[node][succ][key]['action']])
            ys.append(G.node[succ]['state'][state_var])
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    return (model.fit(xs, ys), (xs, ys))
