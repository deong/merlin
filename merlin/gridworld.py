# merlin/gridworld.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merlin, the generator for multitask environments
# for reinforcement learners.
#
# This module contains functions related to generation of gridworld and
# maze-like environments.
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

from __future__ import print_function
import random
import numpy as np
import numpy.random as npr

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
# map the maze onto an internal multigraph representation
#
def graph_from_maze(maze, goals):
    tasks, rows, cols = goals.shape

    # first create all the nodes in a grid
    g = nx.MultiDiGraph()
    for i in range(rows):
        for j in range(cols):
            g.add_node(i*cols + j)

    # now start connecting the nodes, observing walls
    for row in range(rows):
        for col in range(cols):
            source = rowcol_to_index(maze, row, col)
            neighbors = [(x, col) for x in [row-1, row+1]] + [(row, y) for y in [col-1, col+1]]
            for action, (x,y) in enumerate(neighbors):
                target = rowcol_to_index(maze, x, y)

                # if there's a valid target node, add an edge to the target with the
                # corresponding cost. If not, then there's a wall, so add a self-edge
                # with a fixed negative cost.
                if target != None:
                    for task in range(tasks):
                        g.add_edge(source, target, weight=goals[task, x, y])
                else:
                    for task in range(tasks):
                        g.add_edge(source, source, weight=-10)
    return g


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

