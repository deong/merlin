# merlin/io.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merlin, the generator for multitask environments
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
import merlin.gridworld as grd
import cPickle
#from PIL import Image


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
		for index, (_, edge) in enumerate(G.out_edges(node)):
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
# def draw_multimaze(maze, imgx=600, imgy=600):
#     my = len(maze)
#     mx = len(maze[0])
#     
#     image = Image.new("RGB", (imgx, imgy))
#     pixels = image.load()
# 
#     # count number of distinct paths through the maze
#     # note that this counts the walls; hence the -1 on the end
#     m = len(set.union(*[set(row) for row in maze])) - 1
#     
#     color = [(0, 0, 0)] # RGB colors maze paths
#     for i in range(m):
#         color.append((random.randint(0, 255),
#                       random.randint(0, 255),
#                       random.randint(0, 255)))
#         
#     for ky in range(imgy):
#         for kx in range(imgx):
#             pixels[kx, ky] = color[maze[int(my * ky / imgy)][int(mx * kx / imgx)]]
#     image.save(str(m) + "Maze_" + str(mx) + "x" + str(my) + ".png", "PNG")



# write out a trained pybrain neural network that can be read back in
#
# parameters:
#   net: the neural net to save
#   outf: the name of the file to write the network to
#
def write_neural_net(net, trainset, outf):
	nnetFile = open(outf, 'wb')
	cPickle.dump(net, nnetFile)
	cPickle.dump(trainset, nnetFile)
	nnetFile.close()


# read back in a trained neural network and return it
#
# parameters:
#   inf: the name of the input file to read
#   
def read_neural_net(inf):
	nnetFile = open(inf, 'rb')
	net = cPickle.load(nnetFile)
	trainset = cPickle.load(nnetFile)
	return (net, trainset)



# write the dynamics of the trained network along with the "real" dynamics"
#
# parameters:
#   nnet: a trained neural network predicting the transition dynamics
#   training_set: a set of input/output pairs specifying the target dynamics
#   outf: the name of a file to write the results to
#   
def write_train_log(nnet, training_set, outf):
	dynfile = open(outf, 'w')
	for inp, target in training_set:
		approx = nnet.activate(inp)
		entry = inp.tolist() + target.tolist() + approx.tolist()
		dynfile.write("{}\n".format(" ".join([str(x) for x in entry])))
	dynfile.close()
		


# write the dynamics of the trained network along with the "real" dynamics"
#
# parameters:
#   nnet: a trained neural network predicting the transition dynamics
#   training_set: a set of input/output pairs specifying the target dynamics
#   outf: the name of a file to write the results to
#   
def write_svm_train_log(models, training_sets, outf):
	dynfile = open(outf, 'w')
	outputs = {}
	targets = {}
	for task, (inp, outp) in enumerate(training_sets):
		out = []
		tar = []
		for i in range(len(inp)):
			out.append(models[task].predict(inp[i])[0])
			tar.append(outp[i])
		outputs[task] = out
		targets[task] = tar

	# now we hae outputs and targets as tasksXinputs matrices of data
	for task, (inp, outp) in enumerate(training_sets):
		for i in range(len(inp)):
			line = inp[i].tolist()
			for j in range(len(targets)):
				line += [targets[j][i]]
			for j in range(len(outputs)):
				line += [outputs[j][i]]
			dynfile.write("{}\n".format(" ".join([str(x) for x in line])))
	dynfile.close()
		


# write out a series of trained support vector regression models with training data
#
# parameters:
#   models: list of trained SVR models
#   datasets: list of lists of [x, y] training data sets
#   outf: the name of the file to write the models to
#   
def write_svm_model(models, datasets, outf):
	svmFile = open(outf, 'wb')
	cPickle.dump(models, svmFile)
	cPickle.dump(datasets, svmFile)
	svmFile.close()



# read in a series of trained svm models with training data
#
# parameters:
#   inf: the name of the input file to read
#
def read_svm_model(inf):
	svmFile = open(inf, 'rb')
	models = cPickle.load(svmFile)
	datasets = cPickle.load(svmFile)
	svmFile.close()
	return (models, datasets)



# write out a series of trained gaussian process regression models with training data
#
# parameters:
#   models: list of trained GP models
#   datasets: list of lists of [x, y] training data sets
#   outf: the name of the file to write the models to
#   
def write_gp_model(models, datasets, outf):
	gpFile = open(outf, 'wb')
	cPickle.dump(models, gpFile)
	cPickle.dump(datasets, gpFile)
	gpFile.close()



# read in a series of trained gaussian process models with training data
#
# parameters:
#   inf: the name of the input file to read
#
def read_gp_model(inf):
	gpFile = open(inf, 'rb')
	models = cPickle.load(gpFile)
	datasets = cPickle.load(gpFile)
	gpFile.close()
	return (models, datasets)



# write the given graph and annotations out to a file suitable for graphing with graphviz
#
# parameters:
#   G: the graph to output
#   outputfile: the name of the graphiz (dot) file to output to
#   
def output_dot(G, outputfile):
	f = open(outputfile, 'w')
	f.write('digraph mdp {\n')
	f.write('rankdir=LR;\n')
	# f.write('rotate=90;\n')

	# first write the nodes
	for i, node in enumerate(G):
		vals = [round(x, 3) for x in G.node[node]['state']]
		label = str(vals)
		if i == 0:
			f.write('{} [label=\"{}\", shape=box];\n'.format(i, label))
		else:
			f.write('{} [label=\"{}\"];\n'.format(i, label))

	# and then the edges
	for i, node in enumerate(G):
		for j, (_, succ, key) in enumerate(G.out_edges(node, keys=True)):
			f.write('{} -> {} [label=\"{:.3f}\"];\n'.format(i, j, G.edge[node][succ][key]['action']))

	f.write('}\n')
	f.close()
										   
