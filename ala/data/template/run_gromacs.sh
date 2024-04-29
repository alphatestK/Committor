#!/bin/bash

### compulsory ###
ncore=1
tprfile=input.sA.tpr
gmx=`which gmx_mpi`

### optional ###
nsteps=$[500*1000*10] #last is ns
ntomp=4
maxh=48:00 #h:min
filename=alanine
plumedfile=plumed.dat
extra_cmd=""
gpu_id=1

### setup ###
[ -z "$filename" ]  && filename=simulation
outfile=${filename}.out
[ -z "$plumedfile" ] || plumedfile="-plumed $plumedfile"
[ -z "$ntomp" ] || ntomp="-ntomp $ntomp"
[ -z "$nsteps" ] || nsteps="-nsteps $nsteps"
if [ ! -z "$maxh" ]
then
  maxh=`python <<< "print('%g'%(${maxh%:*}+${maxh#*:}/60))"`
  maxh="-maxh $maxh"
fi

### commands ###
mpi_cmd="$gmx mdrun -s $tprfile -deffnm $filename $plumedfile $ntomp $nsteps $maxh -gpu_id $gpu_id "
submit="time mpirun -np $ncore ${mpi_cmd} -pin off" #change this when submitting to a cluster

### execute ###
bck.meup.sh -i $outfile
bck.meup.sh -i ${filename}* > $outfile
echo -e "\n$submit &>> $outfile"
eval "$submit &>> $outfile"
[ -z "$extra_cmd" ] || eval $extra_cmd

