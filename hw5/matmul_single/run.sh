#!/bin/bash

srun --nodes=1 --exclusive --gres=gpu:1 numactl --physcpubind 0-31 ./main $@
