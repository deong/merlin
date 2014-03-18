#!/bin/bash
#
# plots a random walk along a given environment
#
# usage: ./plot_approx_dynamics.sh <walk_log> <output_file> <state_dim> <reward_dim>
#
# where
#     walk_log:     a file written out with the --write-log or --write-rlog options to MERLIN
#     output_file:  either the name to save a PDF to or "term" to plot interactively
#     state_dim:    dimensionality of the state space (equal to number of inputs to the network minus one
#     reward_dim:   dimensionality of the output vector from the neural net
#     

if [ $# -ne 4 ]; then
	echo "usage: plot_walk.sh <walk_file> <output_file> <state_dim> <reward_dim>"
	exit 1
fi

DATA_FILE=$1
OUTPUT_FILE=$2
INPUTD=$3
OUTPUTD=$4

GPFILE=$(mktemp)


# plot the walk through the state space
if [ "$OUTPUT_FILE" != "term" ]; then
	echo "set terminal pdf" >> $GPFILE
	echo "set output \"states-$OUTPUT_FILE\"" >> $GPFILE
fi
echo "set multiplot layout $INPUTD,1" >> $GPFILE
for (( statevar=1; statevar<=$INPUTD; statevar++ )); do
	echo "plot \"$DATA_FILE\" using $statevar with lines title \"s_$statevar\"" >> $GPFILE
done
echo "unset multiplot" >> $GPFILE
cat $GPFILE | gnuplot -p
rm $GPFILE


# and then through the reward space
if [ "$OUTPUT_FILE" != "term" ]; then
	echo "set terminal pdf" >> $GPFILE
	echo "set output \"rewards-ls$OUTPUT_FILE\"" >> $GPFILE
fi
echo "set multiplot layout $OUTPUTD,1" >> $GPFILE
offset=$((INPUTD*2+1))
for (( rvar=1; rvar<=$OUTPUTD; rvar++ )); do
	echo "plot \"$DATA_FILE\" using $((rvar+offset)) with lines title \"R_$rvar\"" >> $GPFILE
done

echo "unset multiplot" >> $GPFILE

cat $GPFILE | gnuplot -p
rm $GPFILE

