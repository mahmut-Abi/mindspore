#!/bin/bash

# 单机多卡

EXEC_PATH=$(pwd)
if [ ! -d "${EXEC_PATH}/MNIST_Data" ]; then
    if [ ! -f "${EXEC_PATH}/MNIST_Data.zip" ]; then
        wget http://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip
    fi
    unzip MNIST_Data.zip
fi
export DATA_PATH=${EXEC_PATH}/MNIST_Data/train/

rm -rf msrun_log
mkdir msrun_log
echo "start training"

# 使用msrun指令在当前节点拉起1个Scheduler进程以及8个Worker进程（无需设置master_addr，默认为127.0.0.1；单机无需设置node_rank）：

msrun --worker_num=8 --local_worker_num=8 --master_port=8118 --log_dir=msrun_log --join=True --cluster_time_out=300 msrun_launcher.py
