# generator.jl
#
# Problem generator for multi-task reinforcement learning problems
#
# Usage:
#
#
# License:
# 
# Copyright 2012 Deon Garrett <deong@cataclysmicmutation.com>
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



# create an instance of a multi-task reinforcement learning problem
#
# parameters:
#   states:  number of states
#   actions: number of actions
#   tasks:   number of tasks
#   covmat:  txt covariance matrix 
#
# Generates a set of reward values, one for each (s,a,t) triplet,
# where s\in {States}, a\in {Actions}, t\in {Tasks}, such that the
# values of the rewards for (s,a,t_i) and (s,a,t_j) are correlated
# with coefficient covmat(i,j)
#
# Note that it is in general not possible to generate variables with
# arbitrary correlations with multiple additional variables. What this
# does is take the first non-zero correlation coefficient and use that.
function make_instance(states, actions, covmat)
    tasks = rank(covmat)
    rewards = Array(Float64, states, actions, tasks)

    mean_vals = zeros(1,tasks)
    N = states*actions
    vals = randmvn(mean_vals, covmat, N)
    
    # note that at this point, vals contains the complete set of
    # state/action values, each column of which is correctly correlated.
    # What's left is to reshape each column into a SxA matrix to yield
    # the final TxSxA dataset
    for task = 1:size(vals,2)
        rewards[:,:,task] = reshape(vals[:,task], states, actions)
    end
    return rewards
end


# generate a sample from a multivariate normal distribution
#
# parameters:
#   mu: mean of each variable
#   R:  covariance matrix
#   N:  number of points to sample
#
# returns:
#   an Nxk matrix (where k=rank(r)) such that each row represents
#   one k-dimensional point and the columns of each of the N such
#   points are inter-correlated according to the specified covariance
#   matrix.
function randmvn(mu, R, N)
    v = mu
    # make sure mu is a column vector
    m, n = size(mu)
    c = max(m, n)
    if n == c
        v = mu'
    end
    
    # make sure R is a valid covariance matrix
    m, n = size(R)
    if m != n || m != c
        println("Invalid covariance matrix $R")
        return
    end

    #T = chol(R)
    T = R
    v = repmat(v, 1, N)
    
    D = (randn(N, c) * T)' + v
    return D'
end



# test the correlations of a generated instance
#
# parameters:
#   D: the generated problem instance
#
# returns:
#   the pearson correlation coefficient between each pair of tasks
#
# The correlations should be between tasks. D is NxMxk, where N is
# the number of states, M the number of actions, and k the number of
# tasks. Thus, we want to extract each NxM submatrix, unroll it into
# a column vector, and compare all k such column vectors for their
# correlation coefficients.
function get_corr(D)
    N, M, k = size(D)
    temp = Array(Float64, N*M, k)
    for i = 1:k
        temp[:,i] = reshape(D[:,:,i], N, M)
    end
    return cor_pearson(temp)
end




# print a generated instance to stdout as a readable data file
#
# parameters:
#   D: the generated problem instance
#
# Basic format:
#
# N M k
#
# NxM matrix of rewards for task 1
#
# NxM matrix of rewards for task 2
# ...
# NxM matrix of rewards for task k
#
# kxk covariance matrix
function print_instance(file, D)
    N, M, k = size(D)
    println("$N $M $k\n")

    for task = 1:k
        for state = 1:N
            for action = 1:M
                printf("%10.6f", D[state,action,task])
            end
            printf("\n")
        end
        printf("\n")
    end
    printf("\n")
    
    cov = get_corr(D)
    println("R = ")
    for i = 1:k
        for j = 1:k
            printf("%7.4f", cov[i,j])
        end
        printf("\n")
    end
end


function main()
    NUM_STATES  = 100
    NUM_ACTIONS = 5
    COV_MAT = [ 1.0  0.4  -0.4;
                0.4  1.0   0.6;
               -0.4  0.6   1.0 ]
    
    instance = make_instance(NUM_STATES, NUM_ACTIONS, COV_MAT)
    print_instance("instance.dat", instance)
end
