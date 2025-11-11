#!/bin/bash

srun --nodes=1 --exclusive --gres=gpu:4 numactl --physcpubind 0-31 ./main $@
