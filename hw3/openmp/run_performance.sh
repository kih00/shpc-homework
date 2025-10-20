#!/bin/bash

srun --nodes=1 --exclusive numactl --physcpubind 0-31 ./main -v -t 32 -n 10 4096 4096 4096
