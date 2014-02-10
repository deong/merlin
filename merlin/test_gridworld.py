#!/usr/bin/env python
#
# merlin/test_gridworld.py
#
# Copyright (c) 2014 Deon Garrett <deon@iiim.is>
#
# This file is part of merlin, the generator for multitask environments
# for reinforcement learners.
#
# Unit tests for the multi-task reinforcement learning problem generator
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

import unittest
import merlin.rewards as rewards
import merlin.gridworld as grid
import numpy as np

# Test cases for maze generation
class TestMultimaze1(unittest.TestCase):
    def setUp(self):
        self.R = np.asarray([[ 1.0,  0.4, -0.4],
                             [ 0.4,  1.0,  0.6],
                             [-0.4,  0.6,  1.0]])
        self.mu = [100.0] * 3
        self.sigma = [10.0] * 3
        self.cov = rewards.cor2cov(self.R, self.sigma)
        self.maze = grid.make_multimaze(4, 4, 3)
        self.goals = grid.maze_goal_states(self.maze, 3, self.mu, self.cov)

    # make sure that each task has a positive goal state somewhere
    def test_goals_exist(self):
        tasks, rows, cols = self.goals.shape
        for task in range(tasks):
            # find the positive goal state for the current task
            goal_loc = None
            for row in range(rows):
                for col in range(cols):
                    if self.goals[task, row, col] > 0:
                        goal_loc = grid.rowcol_to_index(self.maze, row, col)
            self.assertTrue(goal_loc != None)
 

if __name__ == '__main__':
    unittest.main()

