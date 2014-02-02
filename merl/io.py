# merl/io.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merl, the generator for multitask environments
# for reinforcement learners.
#
# This module contains functions for reading and writing the generated
# problems.
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
import random
import merl.gridworld as grd
from PIL import Image


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
#	G: state transition graph
#	R: reward structure
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
			node_num = grd.rowcol_to_index(maze, row, col)
			line = "{} ".format(node_num)

			# order of actions is up, down, left, right
			neighbors = [(x, col) for x in [row-1, row+1]] + [(row, y) for y in [col-1, col+1]]
			for action, (x,y) in enumerate(neighbors):
				target = grd.rowcol_to_index(maze, x, y)
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



# draw a maze graphically with shading indicate paths to different goals
#
# parameters:
#   maze: the input maze
#   imgx: the horizontal size of the output file
#   imgy: the vertical size of the output file
#   
def draw_multimaze(maze, imgx=600, imgy=600):
    my = len(maze)
    mx = len(maze[0])
    
    image = Image.new("RGB", (imgx, imgy))
    pixels = image.load()

    # count number of distinct paths through the maze
    # note that this counts the walls; hence the -1 on the end
    m = len(set.union(*[set(row) for row in maze])) - 1
    
    color = [(0, 0, 0)] # RGB colors maze paths
    for i in range(m):
        color.append((random.randint(0, 255),
                      random.randint(0, 255),
                      random.randint(0, 255)))
        
    for ky in range(imgy):
        for kx in range(imgx):
            pixels[kx, ky] = color[maze[int(my * ky / imgy)][int(mx * kx / imgx)]]
    image.save(str(m) + "Maze_" + str(mx) + "x" + str(my) + ".png", "PNG")
