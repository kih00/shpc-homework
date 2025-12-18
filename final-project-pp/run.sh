#!/bin/bash

: ${NODES:=4}

salloc -N $NODES -x a08,a09 --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		./main $@
