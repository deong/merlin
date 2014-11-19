MERLIN
========

Merlin is the Multi-task Environments for Reinforcement LearnINg generator.

## Building

Requires python2 and the following packages:
- numpy
- scipy
- networkx
- scikit-learn
- pybrain (temporarily)
- pygraphviz if you want to use Dot to visualize graphs

Merlin should work under python3 as well, but many of the required packages are
hard to find and get working in python3, so python2.7 current gets the most
testing.

PyGraphviz is unsupported on Windows. You could try the unofficial installers
here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#pygraphviz
If that doesn't work, see the following websites:
http://stackoverflow.com/questions/2798858/installing-pygraphviz-on-windows-python-2-6
http://stackoverflow.com/questions/4571067/installing-pygraphviz-on-windows/7537047
https://networkx.lanl.gov/trac/ticket/491
https://networkx.lanl.gov/trac/ticket/117

## Usage

Running `merlin.py -h` will display a help message, which looks like
the following:

    > ./merlin.py --help
    usage: merlin.py [-h] [-t {discrete,continuous,perturbation,maze}]
                     [--graph-type {random,maze,lobster}]
                     [--model-type {nnet,svm,gp}]
                     [--landscape-type {fractal,gaussian}] [-n STATES]
                     [-m ACTIONS] [-k TASKS] [-c CORRELATION] [-r RMEANS]
                     [-s STDEV] [-x ROWS] [-y COLS] [--ruggedness RUGGEDNESS]
                     [--landscape-scale LANDSCAPE_SCALE] [-d DIMENSIONS]
                     [--hidden HIDDEN] [--rhidden RHIDDEN]
                     [--transitions-net TRANSITIONS_NET]
                     [--rewards-net REWARDS_NET]
                     [--transitions-dot TRANSITIONS_DOT]
                     [--transitions-log TRANSITIONS_LOG]
                     [--rewards-log REWARDS_LOG] [--max-epochs MAX_EPOCHS]
                     [--baseline BASELINE] [--fuzz-frac FUZZ_FRAC]
                     [--fuzz-scale FUZZ_SCALE] [--svm-C SVM_C]
                     [--svm-epsilon SVM_EPSILON] [--theta0 THETA0]
    
    optional arguments:
      -h, --help            show this help message and exit
      -t {discrete,continuous,perturbation,maze}, --type {discrete,continuous,perturbation,maze}
                            type of generated instance
      --graph-type {random,maze,lobster}
                            graph generation algorithm for state transition
                            function
      --model-type {nnet,svm,gp}
                            generative model type for continuous problems
      --landscape-type {fractal,gaussian}
                            type of model for mapping values onto states
      -n STATES, --states STATES
                            size of the state space
      -m ACTIONS, --actions ACTIONS
                            number of available actions
      -k TASKS, --tasks TASKS
                            number of concurrent tasks
      -c CORRELATION, --correlation CORRELATION
                            task correlation matrix (in Python nested list form)
      -r RMEANS, --rmeans RMEANS
                            mean values for each task reward (in python list form)
      -s STDEV, --stdev STDEV
                            standard deviations of task rewards (in python list
                            form)
      -x ROWS, --rows ROWS  rows in random maze
      -y COLS, --cols COLS  columns in random maze
      --ruggedness RUGGEDNESS
                            ruggedness of the state-transition functions in
                            continuous models
      --landscape-scale LANDSCAPE_SCALE
                            scale factor for state-value functions
      -d DIMENSIONS, --dimensions DIMENSIONS
                            dimensionality of the state vector
      --hidden HIDDEN       number of hidden units in the approximation network
      --rhidden RHIDDEN     number of hidden units in the reward approximation
                            network
      --transitions-net TRANSITIONS_NET
                            file to save the transition dynamics network to
      --rewards-net REWARDS_NET
                            file to save the rewards network to
      --transitions-dot TRANSITIONS_DOT
                            name of the file to write the transition dynamics to
                            in dot format
      --transitions-log TRANSITIONS_LOG
                            name of the file to write training data and
                            predictions to
      --rewards-log REWARDS_LOG
                            name of the file to write reward network training log
                            to
      --max-epochs MAX_EPOCHS
                            maximum number of training epochs for the networks
      --baseline BASELINE   filename containing a trained dynamics network
      --fuzz-frac FUZZ_FRAC
                            fraction of network weights to alter
      --fuzz-scale FUZZ_SCALE
                            amount to alter the chosen network weights by
      --svm-C SVM_C         penalty parameter for the SVM error term
      --svm-epsilon SVM_EPSILON
                            tolerance within which no error is applied
      --theta0 THETA0       default parameters of the autocorrelation model
    

This is quite a volume of options, but most are not required for normal uses.
The most important parameters are the `type` argument, which specifies what
type of problem instance you want to generate. The currently supported options
are as follows.

* `discrete` -- generates finite discrete MDPs based on random graphs for the
  transition functions.
  
* `maze` -- random maze-like problems characterized by the number of rows and
  columns in the maze.
  
* `continuous` -- fits a continuous regression model to represent true
  continuous state and action space MDPs.
  
* `perturbation` -- starts with a previously generated and saved MDP and makes
  small random changes to the transition and reward functions.


For the `discrete` and `continuous` problem types, an underlying random graph
structure is used to represent the transition function (for the continuous
problems, this graph is used to generate training data for a continuous function
approximation scheme). The supported graph types are currently

* `random` -- a random graph with *n* nodes and exactly *m* outgoing edges for
  each node.
  
* `lobster` -- based on a random lobster graph with a "backbone" of *n* nodes
  with additional nodes off the backbone that do not reach the goal state.
  
  
The other obvious parameters are `states`, `actions`, and `tasks`, with the
obvious interpretations regarding the size of the generated problem instances.
For `maze` instances, these parameters are replaced by the `rows` and `cols`
parameters. 

If `tasks` is greater than 1, you can specify the intertask correlation matrices
and mean and variance of the rewards for each task.

For generating continuous MDPs, several additional parameters need to be
considered. Each node and edge in the underlying transition graph must be mapped
onto real-valued state vectors and action values, and the resulting transitions
will be learned by a generative regression model. The `landscape-type` parameter
specifies how successive state values of nodes encountered via random walks
through the state space will be assigned, and a `ruggedness` parameter allows
tuning of (roughly speaking) the autocorrelation along these walks. Informally
speaking, high ruggedness means that the jump in state values as the agent takes
an action will be less predictable. The dimensionality of these generated state
vectors is specified by the `dimensions` parameter.

The `model-type` parameter controls what type of generative model will be used
to learn this state transition function. Each of these models (currently neural
networks, support vector regression, and gaussian process regression) can take
independent parameters such as the number of hidden units in the autoencoding
network or the $C$ and $epsilon$ parameters for the SVM regression.

Finally, there are a number of parameters governing the output of the generated
problem instance. Currently, discrete and maze instances are simply written to
stdout. Continuous models require a more complex strategy. The
`transitions-model` and `rewards-model` parameters give the names of files where
the learned models will be stored (using python's pickle format). You may also
want to record the output of the training pass of each using the
`transitions-log` and `rewards-log` parameters, but these are only necessary to
see what the generator is doing. They are not required to use the generated
instances. Finally, the `transitions-dot` parameter allows Merlin to write a
representation of the underlying graph for any of the graph-based instances in
dot notation (can be visualized using Graphviz's `dot` tool).

