#!/bin/python3.11

# 从指定样本索引子序列中随机采样指定数目的样本数据。
# 下面的样例使用子序列随机采样器从CIFAR-10数据集的指定子序列中抽样3个样本，并展示已读取数据的形状和标签。
# 从打印结果可以看到，采样器从索引序列中随机采样了6个样本。

from mindspore.dataset import SubsetRandomSampler
from mindspore.dataset import Cifar10Dataset
from WeightedRandomSampler import plt_result

DATA_DIR = "./cifar-10-batches-bin/"

# 指定样本索引序列
indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sampler = SubsetRandomSampler(indices, num_samples=6)
# 加载数据
dataset = Cifar10Dataset(DATA_DIR, sampler=sampler)

plt_result(dataset, 2)