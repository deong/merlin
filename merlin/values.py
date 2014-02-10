# merlin/values.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merlin, the generator for multitask environments
# for reinforcement learners.
#
# This module defines functions for assigning state and action values
# to nodes in the transition graph.
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
import numpy.random as npr

# return a 1-D fractal landscape generated using the midpoint displacement method
#
# parameters:
#   n: the number of points in the landscape
#   ruggedness: a parameter determining ruggedness [0.0=very rugged, 1.0=very smooth]
#   rng: initial range of the random number generator
#   seed: baseline value of the landscape
# returns:
#   a list of points mapping out landscape heights
#   
def make_fractal(n, ruggedness, rng=1.0, seed=0):
	points = [npr.random(), npr.random()]
	return make_fractal_aux(n, points, ruggedness, rng)

# helper function for the fractal generation
def make_fractal_aux(n, points, ruggedness, rng):
	if len(points) >= n:
		return points[:n]
	newpoints = [points[0]]
	for i in range(len(points)-1):
		p1 = points[i]
		p2 = points[i+1]
		newval = (p1+p2)/2 + npr.random()*(rng*2)-rng
		newpoints.append(newval)
		newpoints.append(p2)
	rng *= 2**(-ruggedness)
	return make_fractal_aux(n, newpoints, ruggedness, rng)



# return a length n random walk with each step adding Gaussian noise
#
# parameters:
#   n: the number of points on the walk
#   stdev: the size of the jumps from step to step
#   seed: the initial value along the walk
# returns:
#   a list of points mapping out the walk
#   
def make_gaussian_walk(n, stdev, seed=0):
	ys = []
	for i in range(n):
		y = seed + npr.normal(0, stdev)
		ys.append(y)
		seed = y
	return ys

