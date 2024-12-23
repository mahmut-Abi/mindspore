#!/bin/python3.11

## note


import mindspore as ms
import mindspore.numpy as np
from mindspore import dtype as mstype

# int索引取值
tensor_x = ms.Tensor(np.arange(2 * 3 * 2).reshape((2, 3, 2)))
data_single = tensor_x[0]
data_multi = tensor_x[0][1]
print('int索引取值')
print('data_single:')
print(data_single)
print('data_multi:')
print(data_multi)

# bool索引取值
tensor_x = ms.Tensor(np.arange(2 * 3).reshape((2, 3)))
data_single = tensor_x[True]
data_multi = tensor_x[True][True]
print('bool索引取值')
print('data_single:')
print(data_single)
print('data_multi:')
print(data_multi)

# None索引取值
tensor_x = ms.Tensor(np.arange(2 * 3).reshape((2, 3)))
data_single = tensor_x[...]
data_multi = tensor_x[...][...]
print('None索引取值')
print('data_single:')
print(data_single)
print('data_multi:')
print(data_multi)

# slice索引取值
tensor_x = ms.Tensor(np.arange(4 * 2 * 2).reshape((4, 2, 2)))
data_single = tensor_x[1:4:2]
data_multi = tensor_x[1:4:2][1:]
print('slice索引取值')
print('data_single:')
print(data_single)
print('data_multi:')
print(data_multi)

# Tensor索引取值
tensor_x = ms.Tensor([1, 2, 3])
tensor_index = ms.Tensor([True, False, True], dtype=mstype.bool_)
output = tensor_x[tensor_index]
print('Tensor索引取值')
print(output)

# List索引取值
tensor_x = ms.Tensor(np.arange(4 * 2 * 3).reshape((4, 2, 3)))
list_index0 = [1, 2, 0]
list_index1 = [True, False, True]
data_single = tensor_x[list_index0]
data_multi = tensor_x[list_index0][list_index1]
print('List索引取值')
print('data_single:')
print(data_single)
print('data_multi:')
print(data_multi)

# Tuple索引取值
tensor_x = ms.Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)))
tensor_index = ms.Tensor(np.array([[1, 2, 1], [0, 3, 2]]), mstype.int32)
data = tensor_x[1, 0:1, tensor_index]
print('Tuple索引取值')
print('data:')
print(data)