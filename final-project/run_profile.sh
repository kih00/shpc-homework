#!/bin/bash

: ${NODES:=4}

NOW=$(date +%m%d_%H%M)

# : ${PROFILE:=nsys profile --cudabacktrace=all --capture-range=cudaProfilerApi}

salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		nsys profile --cudabacktrace=all --capture-range=cudaProfilerApi \
		--output report-${NOW}-node%q{OMPI_COMM_WORLD_RANK} ./main $@