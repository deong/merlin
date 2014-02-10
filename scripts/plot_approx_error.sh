#!/usr/bin/gnuplot

# set terminal pdf
# set output 'train_log.pdf'

set multiplot layout 3,1
plot 'train_log.dat' using 5 with lines title 'V1 True', 'train_log.dat' using 8 with lines title 'V1 Approx'
plot 'train_log.dat' using 6 with lines title 'V2 True', 'train_log.dat' using 9 with lines title 'V2 Approx'
plot 'train_log.dat' using 7 with lines title 'V3 True', 'train_log.dat' using 10 with lines title 'V3 Approx'
unset multiplot
