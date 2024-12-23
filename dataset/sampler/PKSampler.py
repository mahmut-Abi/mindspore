#!/bin/python3.11

## note
# 在指定的数据集类别P中，每种类别各采样K条数据。
# 下面的样例使用PK采样器从CIFAR-10数据集中每种类别抽样2个样本，最多10个样本，并展示已读取数据的形状和标签。
# 从打印结果可以看出，采样器对数据集中的每种标签都采样了2个样本，一共10个样本。

from mindspore.dataset import PKSampler
from mindspore.dataset import Cifar10Dataset
from WeightedRandomSampler import plt_result

DATA_DIR = "./cifar-10-batches-bin/"

# 每种类别抽样2个样本，最多10个样本
sampler = PKSampler(num_val=2, class_column='label', num_samples=10)
dataset = Cifar10Dataset(DATA_DIR, sampler=sampler)

plt_result(dataset, 3)
