#!/bin/bash

# source programs
source /home/pkang@iit.local/program/plumed/plumed2-dev/sourceme.sh

# define core offset and number
CORE_NUM=3
CORE_OFFSET=0


for i in 2 4 8 10
	do
		LAMBDA=$i
		FOLDER_NAME="lambda_${LAMBDA}" 
		echo folder $FOLDER_NAME

		# create folder from template (need inputs, plumed.dat and model)
		cp -r ../template $FOLDER_NAME

		# move to folder
		cd $FOLDER_NAME

		# modify lambda in plumed.dat
		sed -i "s/LAMBDA=lambda/LAMBDA=$LAMBDA/g" plumed.dat


		# cd A + run sim A
		cd I
		bash ../run_cp2k_sim.sh $CORE_OFFSET-$(($CORE_OFFSET+$CORE_NUM-1)) &
		cd ..
		CORE_OFFSET=$(($CORE_OFFSET+$CORE_NUM))
		
		# cd B + run sim B
		cd P
		bash ../run_cp2k_sim.sh $CORE_OFFSET-$(($CORE_OFFSET+$CORE_NUM-1)) &
		CORE_OFFSET=$(($CORE_OFFSET+$CORE_NUM))
	
		cd ..
		cd ..
	done
