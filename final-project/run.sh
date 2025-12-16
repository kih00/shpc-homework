#!/bin/bash

: ${NODES:=4}
# : ${PROFILE:=nsys profile --cudabacktrace=all}

salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		$PROFILE ./main $@
