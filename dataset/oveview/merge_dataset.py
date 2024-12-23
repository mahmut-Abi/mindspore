#!/bin/python3.11

import mindspore.dataset as ds

# 设置随机种子以确保可重复性
ds.config.set_seed(1234)

# 定义两个数据集
data = [1, 2, 3]
dataset1 = ds.NumpySlicesDataset(data=data, column_names=["column_1"])

data = [4, 5, 6]
dataset2 = ds.NumpySlicesDataset(data=data, column_names=["column_1"])

# 以 concat 的形式串联
dataset_concat = dataset1.concat(dataset2)
print("串联结果:")
for item in dataset_concat.create_dict_iterator():
    print(item)

######
# 定义两个数据集
data = [1, 2, 3]
dataset1 = ds.NumpySlicesDataset(data=data, column_names=["column_1"])

data = [4, 5, 6]
dataset2 = ds.NumpySlicesDataset(data=data, column_names=["column_2"])
# 以zip 方法并联
print("并联结果:")
dataset_zip = dataset1.zip(dataset2)
for item in dataset_zip.create_dict_iterator():
    print(item)
