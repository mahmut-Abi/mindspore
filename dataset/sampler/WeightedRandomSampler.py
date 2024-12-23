#!/bin/python3.11

## note
# 指定长度为N的采样概率列表，按照概率在前N个样本中随机采样指定数目的数据。
# 下面的样例使用带权随机采样器从CIFAR-10数据集的前10个样本中按概率获取6个样本，并展示已读取数据的形状和标签。
# 从打印结果可以看出，本次在前面一共10个样本中随机采样了6条数据，只有前面两个采样概率不为0的样本才有机会被采样。

import math
import matplotlib.pyplot as plt
from mindspore.dataset import WeightedRandomSampler, Cifar10Dataset

DATA_DIR = "./cifar-10-batches-bin/"

# 指定前10个样本的采样概率并进行采样
weights = [0.8, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]
sampler = WeightedRandomSampler(weights, num_samples=6)
dataset = Cifar10Dataset(DATA_DIR, sampler=sampler)  # 加载数据

def plt_result(dataset, row):
    """显示采样结果"""
    num = 1
    for data in dataset.create_dict_iterator(output_numpy=True):
        print("Image shape:", data['image'].shape, ", Label:", data['label'])
        plt.subplot(row, math.ceil(dataset.get_dataset_size() / row), num)
        image = data['image']
        plt.imshow(image, interpolation="None")
        num += 1

plt_result(dataset, 2)
