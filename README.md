mtlgen
========

Random problem generator for multi-task reinforcement learning problems.

## Building

Requires python3 and the following packages:
- numpy
- scipy
- networkx

Note that a python3 compatible networkx may be tough to find for your
distribution of Linux. There is a debian package available in experimental
though which works fine.


## Usage

Running `python mtlgen.py -h` will display a help message, which looks like
the following:

    usage: mtlgen.py [-h] [--demo] [-t TYPE] [-n STATES] [-m ACTIONS] [-k TASKS]
                     [-c CORRELATION] [-s STDEV] [-x ROWS] [-y COLS]
    
    optional arguments:
      -h, --help            show this help message and exit
      --demo                run with a sample set of parameters
      -t TYPE, --type TYPE  problem instance type {rgudcr,rzcgl}
      -n STATES, --states STATES
                            size of the state space
      -m ACTIONS, --actions ACTIONS
                            number of available actions
      -k TASKS, --tasks TASKS
                            number of concurrent tasks
      -c CORRELATION, --correlation CORRELATION
                            task correlation matrix (in Python nested list form)
      -s STDEV, --stdev STDEV
                            standard deviations of task rewards (in python list
                            form)
      -x ROWS, --rows ROWS  rows in random maze
      -y COLS, --cols COLS  columns in random maze


The most important parameters are the `type` argument, which specifies what
type of problem instance you want to generate. The currently supported options
are as follows.

* `rgudcr` -- random graph, uniform out-degree, correlated rewards. 
  This produces problem instances with a given number of states, each of which
  has exactly the given number of actions available. The problem will have a
  specified number of separate tasks, and the rewards for each task are
  real-valued and distributed throughout the entire state-action space, with
  values that are correlated according to the specified parameters for
  correlation and standard deviation.
  
* `rzcgl` -- random maze-like problems, correlated goal locations.
  These problems are basic 2D gridworld problems with K separate goal states,
  one for each specified task. The locations of the goal states are related to
  one another through the specified correlation matrix.
  

