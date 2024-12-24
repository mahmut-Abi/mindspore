#!/bin/bash

export DATA_PATH=/MNIST_Data/train/

mkdir msrun_log
echo "start training"

msrun --worker_num=8 --local_worker_num=2 --master_addr=<node_1 ip address> --master_port=8118 --node_rank=0 --log_dir=msrun_log --join=True --cluster_time_out=300 train.py