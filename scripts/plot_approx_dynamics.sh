#!/bin/bash

if [ $# -ne 2 ]; then
	echo "usage: plot_approx_dynamics.sh <train.log> <output.pdf>"
	exit 1
fi

DATA_FILE=$1
OUTPUT_FILE=$2

cat <<EOF | gnuplot
set terminal pdf
set output "$OUTPUT_FILE"

set multiplot layout 3,1
plot "$DATA_FILE" using 5 with lines title 'V1 True', "$DATA_FILE" using 8 with lines title 'V1 Approx'
plot "$DATA_FILE" using 6 with lines title 'V2 True', "$DATA_FILE" using 9 with lines title 'V2 Approx'
plot "$DATA_FILE" using 7 with lines title 'V3 True', "$DATA_FILE" using 10 with lines title 'V3 Approx'
unset multiplot
EOF

