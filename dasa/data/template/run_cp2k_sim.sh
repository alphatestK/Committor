#/bin/bash

CORES=$1

source ~/Bin/dev/cp2k-8.1/sourceme.sh

taskset --cpu-list $CORES cp2k.ssmp -i pm6.inp