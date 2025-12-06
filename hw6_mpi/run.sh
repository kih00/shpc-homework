#!/bin/bash

: ${NODES:=4}

salloc -p class1 -N $NODES --exclusive --gres=gpu:4            \
    mpirun --bind-to none -mca btl ^openib -npernode 1         \
    numactl --physcpubind 0-31                                 \
    ./main $@
