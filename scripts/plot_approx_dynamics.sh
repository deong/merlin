#!/bin/bash
#
# plots the approximation error for each transition in a training data set
#
# usage: ./plot_approx_dynamics.sh <train_data_file> <output_file> <input_dim> <output_dim>
#
# where
#     train_data_file: a file written out with the --write-log or --write-rlog options to MERLIN
#     output_file:     either the name to save a PDF to or "term" to plot interactively
#     input_dim:       dimensionality of the state space (equal to number of inputs to the network minus one
#     output_dim:      dimensionality of the output vector from the neural net
#     

if [ $# -ne 4 ]; then
	echo "usage: plot_approx_dynamics.sh <train.log> <output.pdf> <input_dim> <output_dim>"
	exit 1
fi

DATA_FILE=$1
OUTPUT_FILE=$2
INPUTD=$3
OUTPUTD=$4

GPFILE=$(mktemp)

if [ "$OUTPUT_FILE" != "term" ]; then
	echo "set terminal pdf" >> $GPFILE
	echo "set output \"$OUTPUT_FILE\"" >> $GPFILE
fi

echo "set multiplot layout ${OUTPUTD},1" >> $GPFILE

X1=$((INPUTD+2))
X2=$((X1+OUTPUTD))
for (( OUTPUT=1; OUTPUT<=$OUTPUTD; OUTPUT++)); do
	T1="V${OUTPUT} True"
	T2="V${OUTPUT} Approx"
	echo "plot \"$DATA_FILE\" using $X1 with lines title \"$T1\", \"$DATA_FILE\" using $X2 with lines title \"$T2\"" >> $GPFILE
	X1=$((X1+1))
	X2=$((X2+1))
done
echo "unset multiplot" >> $GPFILE

cat $GPFILE | gnuplot -p
rm $GPFILE

