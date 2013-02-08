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

from PIL import Image
import random

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


    
def make_multimaze(width, height):
    # Multi-Maze Generator using Depth-first Search
    # Multi-Maze: Maze w/ multiple paths to solve
    # http://en.wikipedia.org/wiki/Maze_generation_algorithm
    # FB - 20121214
    m = random.randint(5, 15) # of maze paths

    # width and height of the maze
    mx = width  
    my = height 

    # 4 directions to move in the maze
    dx = [0, 1, 0, -1] 
    dy = [-1, 0, 1, 0] 

    maze = [[0 for x in range(mx)] for y in range(my)]

    stack = [] # array of stacks
    for i in range(m):
        while True:
            kx = random.randint(0, mx - 1); ky = random.randint(0, my - 1)
            if maze[ky][kx] == 0: break
        stack.append([(kx, ky)])
        maze[ky][kx] = i + 1

    cont = True # continue
    while cont:
        cont = False
        for p in range(m):
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
    return maze



if __name__ == "__main__":
    maze = make_multimaze(200, 200)
    draw_multimaze(maze, 800, 800)
