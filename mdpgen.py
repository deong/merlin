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


import sys
import argparse
import ast
import random
import numpy as np
import scipy.linalg as sla
import numpy.random as npr
import networkx as nx


#
# create random reward structure for an (nstates, nactions) MDP
#
# parameters:
#   nstates:  number of states
#   nactions: number of actions
#   covmat:   nxn covariance matrix, where n=number of tasks
# returns:
#   an (nstates x nactions x ntasks) reward matrix
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
#   nstates: number of states
#   nactions: number of actions
# returns:
#   a directed multigraph with each node having out-degree exactly
#   equal to edges and in-degree > 0.
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
                    line += "{0:.3f} ".format(R[node, index, task])
        print(line)
    print("\n")


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
    # for each task, build a list of maze locations with that task
    # id
    rows = maze.shape[0]
    cols = maze.shape[1]
    reward_values = npr.multivariate_normal(mu, cov)
    goals = np.zeros([rows, cols, tasks])
    pgoals = np.zeros((rows, cols))
    for task in range(tasks):
        locs = np.transpose(np.where(maze == (task+1)))
        goal_loc = locs[npr.randint(0, len(locs))]
        print("goal_loc={}".format(goal_loc))
        goals[goal_loc[0], goal_loc[1], task] = reward_values[task]
        pgoals[goal_loc[0], goal_loc[1]] = reward_values[task]
    print(pgoals)
    return goals


#
# convert a generated maze to a graph structure to be written out
#
def maze_transition_graph(maze, goals):
    rows, cols, tasks = goals.shape
    rewards = np.zeros((rows*cols, 4, tasks))
    G = nx.MultiDiGraph()
    nodes = maze.size
    assert(nodes == rows*cols)
    G.add_nodes_from(range(nodes))
    for row in range(rows):
        for col in range(cols):
            node_num = rowcol_to_index(maze, row, col)
            # order of neighbors is up, down, left, right
            neighbors = [(x,col) for x in [row-1,row+1]] + [(row,y) for y in [col-1,col+1]]
            for action, (x,y) in enumerate(neighbors):
                idx = rowcol_to_index(maze, x, y)
                if idx != None:
                    G.add_edge(node_num, idx)#, weight=goals[x,y], task=maze[x,y])
                    print("node={}, idx={}, action={}, maze[{},{}]={}, goals[{},{}]={}".format(node_num, idx, action, x, y, maze[x,y], x, y, goals[x,y]))
                    task_num = maze[x,y]-1
                    rewards[node_num, action, task_num] = goals[x,y,task_num]
                else:
                    G.add_edge(node_num, node_num)#, weight=-10, task=0)
                    for task_num in range(tasks):
                        rewards[node_num, action, task_num] = -10
    return (G, rewards)


    
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type",       default=None,  help="problem instance type {rgudcr,rzcgl}")
    parser.add_argument("-n", "--states",     default=100,   help="size of the state space", type=int)
    parser.add_argument("-m", "--actions",    default=4,     help="number of available actions", type=int)
    parser.add_argument("-k", "--tasks",      default=2,     help="number of concurrent tasks", type=int)
    parser.add_argument("-c", "--correlation",               help="task correlation matrix (in Python nested list form)")
    parser.add_argument("-s", "--stdev",                     help="standard deviations of task rewards (in python list form)")
    parser.add_argument("-x", "--rows",       default=10,    help="rows in random maze", type=int)
    parser.add_argument("-y", "--cols",       default=10,    help="columns in random maze", type=int)
    
    
    args = parser.parse_args()


    # testing
    R = np.asarray([[ 1.0,  0.4, -0.4],
                    [ 0.4,  1.0,  0.6],
                    [-0.4,  0.6,  1.0]])
    # mu = np.asarray([0,0,0])
    # sigma = np.asarray([1,1,1])
    # cov = np.diag(sigma).dot(R).dot(np.diag(sigma))
    # rewards = mvnrewards(args.states, args.actions, mu, cov)
    # print(rewards)
    # rewards = np.reshape(rewards, [3, args.states, args.actions])
    # print(rewards)
    # sys.exit(1)

    mu = [100.0] * 3
    sigma = [10.0] * 3
    cov = np.diag(sigma).dot(R).dot(np.diag(sigma))
    z = make_multimaze(4, 4, 3)
    goals = maze_goal_states(z, 3, mu, cov)
    print(cov)
    print(z)
    print(goals)
    mtg, rewards = maze_transition_graph(z, goals)
    write_instance(mtg, rewards)
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
    cov = np.diag(args.stdev).dot(args.correlation).dot(np.diag(args.stdev))
        
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
        transition_graph = maze_transition_graph(maze, goals)
        rewards = np.zeros([args.rows * args.cols, 4, args.tasks])
        write_instance(transition_graph, rewards)
        print("# type={}, rows={}, cols={}, correlation={}, stdev={}".
              format(args.type, args.rows, args.col, args.correlation.tolist(), args.stdev.tolist()))
    else:
        print("invalid problem type specified: {}", args.type)
        parser.print_help()
        sys.exit(1)
        
    # R = np.asarray([[ 1.0,  0.4, -0.4],
    #                 [ 0.4,  1.0,  0.6],
    #                 [-0.4,  0.6,  1.0]])
    # R = np.asarray([[ 1.0,  0.4, -0.4,  0.0],
    #                 [ 0.4,  1.0,  0.6, -0.1],
    #                 [-0.4,  0.6,  1.0,  0.2],
    #                 [ 0.0, -0.1,  0.2,  1.0]])
    # R = np.asarray([[1.0, 0.0],
    #                 [0.0, 1.0]])
