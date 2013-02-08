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
# convert a generated maze to a graph structure to be written out
#
def convert_maze_to_instance(maze):
    G = nx.Graph()
    nodes = maze.size
    G.add_nodes_from(range(nodes))
    for row in range(maze.shape[0]):
        for col in range(maze.shape[2]):
            node_num = rowcol_to_index(maze, row, col)
            up_neighbor = rowcol_to_index(maze, row+1, col)
            if up_neighbor:
                G.add_edge(node_num, up_neighbor)
            down_neighbor = rowcol_to_index(maze, row-1, col)
            if down_neighbor:
                G.add_edge(node_num, down_neighbor)
            left_neighbor = rowcol_to_index(maze, row, col-1)
            if left_neighbor:
                G.add_edge(node_num, left_neighbor)
            right_neighbor = rowcol_to_index(maze, row, col+1)
            if right_neighbor:
                G.add_edge(node_num, right_neighbor)
    
            
#
# take a maze, row, and column, and return a node number or None
# if the requested row and column are out of bounds
#
def rowcol_to_index(maze, row, col):
    index = row*maze.shape[0] + col
    if index >= maze.size:
        return None
    else:
        return index

    
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

    maze = make_multimaze(200,200)
    
    print(maze)
