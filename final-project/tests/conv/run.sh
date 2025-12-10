#!/bin/bash

srun --nodes=1 --exclusive --partition=class1 --gres=gpu:1 numactl --physcpubind 0-31 ./main $@ -n 10 32
# srun --nodes=1 --exclusive --partition=class1 --gres=gpu:1 numactl --physcpubind 0-31 nsys profile --cudabacktrace=all ./main $@
