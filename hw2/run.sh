#!/bin/bash

srun --nodes=1 --exclusive numactl --physcpubind 0-63 ./main $@
