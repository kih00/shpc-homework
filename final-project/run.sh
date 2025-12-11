#!/bin/bash

: ${NODES:=1}
: ${INSPECT:=nsys profile --cudabacktrace=all}

salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		$INSPECT ./main $@
