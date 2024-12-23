#!/bin/python3.11

## note
# 在分布式训练中，对数据集分片进行采样。
# 下面的样例使用分布式采样器将构建的数据集分为4片，在分片抽取一个样本，共采样3个样本，并展示已读取的数据。
# 从打印结果可以看出，数据集被分成了4片，每片有3个样本，本次获取的是id为0的片中的样本。

from mindspore.dataset import DistributedSampler
from mindspore.dataset import NumpySlicesDataset

# 自定义数据集
data_source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 构建的数据集分为4片，共采样3个数据样本
sampler = DistributedSampler(num_shards=4, shard_id=0, shuffle=False, num_samples=3)
dataset = NumpySlicesDataset(data_source, column_names=["data"], sampler=sampler)

# 打印数据集
for data in dataset.create_dict_iterator():
    print(data)
