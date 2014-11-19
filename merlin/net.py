# merlin/net.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merlin, the generator for multitask environments
# for reinforcement learners.
#
# This module defines functionality for dealing with pybrain neural networks
# used to approximate transition graphs and reward functions.
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

import numpy as np
import numpy.random as npr
import pybrain.datasets
import pybrain.tools.shortcuts
import pybrain.supervised.trainers


# iterate the model based on the given neural network, state, and action
#
# parameters:
#   net: the trained neural network
#   state: the current state
#   action: the current action
# returns:
#   the output of the network given the state/action pair
# 
def activation(tnet, state, action):
    inp = np.append(state, action)
    return tnet.activate(inp)


# randomly change fuzz_factor percent of the weights of the given network
#
# parameters:
#   net: the trained network to fuzz
#   frac: the percentage of weights to randomly change
#   scale: the amount to change the selected weights by
#   
def fuzz_neural_net(net, frac, scale):
    weights = net.params
    fweights = [x if npr.random() >= frac else npr.normal(x, scale) for x in weights]
    net._setParameters(fweights)
    return net
