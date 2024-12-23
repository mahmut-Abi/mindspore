#!/bin/python3.11

## note
# 从索引序列中随机采样指定数目的数据。
# 下面的样例使用随机采样器，分别从数据集中有放回和无放回地随机采样5个数据，并打印展示。为了便于观察有放回与无放回的效果，这里自定义了一个数据量较小的数据集。
# 从打印结果可以看出，使用有放回采样器时，同一条数据可能会被多次获取；使用无放回采样器时，同一条数据只能被获取一次。

from mindspore.dataset import RandomSampler, NumpySlicesDataset

np_data = [1, 2, 3, 4, 5, 6, 7, 8]  # Dataset

# 从索引序列中随机采样指定数目的数据
# 定义有放回采样器，采样5条数据
sampler1 = RandomSampler(replacement=True, num_samples=5)
dataset1 = NumpySlicesDataset(np_data, column_names=["data"], sampler=sampler1)

print("With Replacement:    ", end='')
for data in dataset1.create_tuple_iterator(output_numpy=True):
    print(data[0], end=' ')

# 定义无放回采样器，采样5条数据
sampler2 = RandomSampler(replacement=False, num_samples=5)
dataset2 = NumpySlicesDataset(np_data, column_names=["data"], sampler=sampler2)

print("\nWithout Replacement: ", end='')
for data in dataset2.create_tuple_iterator(output_numpy=True):
    print(data[0], end=' ')

# 从上面的打印结果可以看出，使用有放回采样器时，同一条数据可能会被多次获取；使用无放回采样器时，同一条数据只能被获取一次。