# merl/__init__.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merl, the generator for multitask environments
# for reinforcement learners.
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

import random
import numpy as np
import scipy.linalg as sla
import numpy.random as npr
import networkx as nx

__all__ = ['graphs', 'values', 'rewards', 'gridworld', 'io']
